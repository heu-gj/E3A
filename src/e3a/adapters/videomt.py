"""Thin E3A adapter around VidEoMT's official video matcher and criterion."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class VidEoMTMaskCriterion(nn.Module):
    """Train clip-consistent entity masks with VidEoMT's upstream implementation.

    E3A predictions:
        mask_logits: [B,T,K,H,W]
        objectness_logits: [B,T,K]

    Canonical per-clip target contract:
        labels: [N] int64, all zero (the single foreground class "entity")
        ids: [N,T] int64, stable non-negative track ID while visible, -1 otherwise
        masks: [N,T,H_gt,W_gt], binary/soft masks; zero wherever ids == -1

    The adapter converts this clip-level schema to the flattened B*T contract
    consumed by VidEoMT. The upstream consistent matcher assigns each GT track
    to a query at its first visible frame and reuses that assignment across the
    clip. The upstream criterion then supervises each frame's mask/class output.
    Per-frame objectness is mean-pooled over observed frames to form a stable
    clip-level entity logit, matching the clip-consistent class assignment.

    Training batches store entity_targets as padded tensors: labels [B,Nmax],
    ids [B,Nmax,T], masks [B,Nmax,T,H,W], and track_counts [B]. This keeps
    targets aligned with the upstream trainer's recursive batch chunker.

    This adapter does not infer dataset IDs, action labels, or visibility rules.
    The dataset adapter must establish those semantics before supplying targets.
    """

    _LOSS_KEYS = ("loss_ce", "loss_mask", "loss_dice")

    def __init__(
        self,
        *,
        frames: int,
        num_points: int,
        loss_weights: Mapping[str, float],
        matching_costs: Mapping[str, float] | None = None,
        eos_coef: float = 0.1,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        matcher: nn.Module | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if frames <= 0:
            raise ValueError("frames must be positive.")
        if num_points <= 0:
            raise ValueError("num_points must be positive for VidEoMT point sampling.")
        if not math.isfinite(eos_coef) or eos_coef <= 0:
            raise ValueError("eos_coef must be finite and positive.")
        if not math.isfinite(oversample_ratio) or oversample_ratio < 1:
            raise ValueError("oversample_ratio must be finite and at least 1.")
        if (
            not math.isfinite(importance_sample_ratio)
            or not 0 <= importance_sample_ratio <= 1
        ):
            raise ValueError("importance_sample_ratio must be in [0, 1].")

        missing = set(self._LOSS_KEYS).difference(loss_weights)
        if missing:
            raise ValueError(f"loss_weights is missing required keys: {sorted(missing)}")
        normalized_weights = {key: float(loss_weights[key]) for key in self._LOSS_KEYS}
        if any(not math.isfinite(value) or value < 0 for value in normalized_weights.values()):
            raise ValueError("loss weights must be finite and non-negative.")
        if not any(normalized_weights.values()):
            raise ValueError("At least one loss weight must be positive.")

        costs = {"cost_class": 1.0, "cost_mask": 1.0, "cost_dice": 1.0}
        if matching_costs is not None:
            unknown = set(matching_costs).difference(costs)
            if unknown:
                raise ValueError(f"Unknown VidEoMT matching costs: {sorted(unknown)}")
            costs.update({key: float(value) for key, value in matching_costs.items()})
        if any(not math.isfinite(value) or value < 0 for value in costs.values()):
            raise ValueError("Matching costs must be finite and non-negative.")
        if not any(costs.values()):
            raise ValueError("At least one VidEoMT matching cost must be positive.")

        if criterion is None:
            if matcher is None:
                matcher_cls, criterion_cls = self._load_upstream_classes()
                matcher = matcher_cls(
                    **costs,
                    num_points=num_points,
                    frames=frames,
                )
            else:
                _, criterion_cls = self._load_upstream_classes()

            criterion = criterion_cls(
                num_classes=1,
                matcher=matcher,
                weight_dict=normalized_weights,
                eos_coef=eos_coef,
                losses=["labels", "masks"],
                num_points=num_points,
                oversample_ratio=oversample_ratio,
                importance_sample_ratio=importance_sample_ratio,
            )
        elif matcher is None:
            matcher = getattr(criterion, "matcher", None)
        elif getattr(criterion, "matcher", matcher) is not matcher:
            raise ValueError("The injected criterion and matcher must use the same matcher.")

        if not isinstance(criterion, nn.Module):
            raise TypeError("criterion must be a torch.nn.Module.")
        if not isinstance(matcher, nn.Module):
            raise TypeError("matcher must be a torch.nn.Module.")

        self.frames = int(frames)
        self.num_points = int(num_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        self.loss_weights = normalized_weights
        self.criterion = criterion
        self._matcher_fallback = matcher if not hasattr(criterion, "matcher") else None

    @staticmethod
    def _register_upstream_package(name: str, package_path: Path) -> None:
        """Register a narrow package path without executing VidEoMT's root __init__."""
        package_path = package_path.resolve()
        package = sys.modules.get(name)
        if package is None:
            package = ModuleType(name)
            package.__path__ = [str(package_path)]
            package.__package__ = name
            sys.modules[name] = package
            return

        existing_paths = {
            Path(path).resolve()
            for path in getattr(package, "__path__", [])
        }
        if package_path not in existing_paths:
            raise ImportError(
                f"{name} is already imported from a different location; "
                "E3A requires the pinned third_party/videomt source."
            )

    @classmethod
    def _load_upstream_classes(cls):
        repo_root = Path(__file__).resolve().parents[3]
        upstream_root = repo_root / "third_party" / "videomt"
        package_root = upstream_root / "videomt"
        if not package_root.is_dir():
            raise ImportError(
                "VidEoMT source is missing. Run "
                "git submodule update --init third_party/videomt."
            )

        # Import only the pinned matcher, criterion, and shared misc module.
        # The upstream root __init__ eagerly imports its full Detectron2 model,
        # dataset, and config stack, none of which E3A needs for these losses.
        cls._register_upstream_package("videomt", package_root)
        cls._register_upstream_package("videomt.modeling", package_root / "modeling")
        cls._register_upstream_package("videomt.utils", package_root / "utils")

        try:
            from videomt.criterion_videomt import VideoSetCriterion
            from videomt.modeling.matcher import VideoHungarianMatcher_Consistent
        except ImportError as exc:
            raise ImportError(
                "VidEoMT matcher/criterion dependencies are required. Initialize "
                "third_party/videomt and install compatible SciPy and Detectron2 "
                "point-sampling utilities in the target PyTorch/CUDA environment."
            ) from exc
        return VideoHungarianMatcher_Consistent, VideoSetCriterion

    @property
    def matcher(self) -> nn.Module:
        upstream_matcher = getattr(self.criterion, "matcher", None)
        return upstream_matcher if upstream_matcher is not None else self._matcher_fallback

    def _unpack_batched_targets(
        self,
        targets: Mapping[str, Any],
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        """Unpack padded targets while preserving per-track annotation coverage."""
        required = {"labels", "ids", "masks", "track_counts"}
        missing = required.difference(targets)
        if missing:
            raise ValueError(
                f"batched entity_targets is missing keys: {sorted(missing)}"
            )

        labels = targets["labels"]
        track_ids = targets["ids"]
        masks = targets["masks"]
        track_counts = targets["track_counts"]
        base_values = (labels, track_ids, masks, track_counts)
        if not all(torch.is_tensor(value) for value in base_values):
            raise TypeError("batched entity_targets values must all be torch tensors.")
        if "mask_valid" in targets:
            mask_valid = targets["mask_valid"]
        elif (
            track_ids.ndim == 3
            and track_counts.ndim == 1
            and track_counts.shape[0] == track_ids.shape[0]
        ):
            # Backward compatibility for old padded batches: real track rows
            # are fully annotated, while collator padding is never supervision.
            row_valid = (
                torch.arange(track_ids.shape[1], device=track_ids.device)[None, :]
                < track_counts.to(device=track_ids.device, dtype=torch.int64)[:, None]
            )
            mask_valid = row_valid[:, :, None].expand_as(track_ids)
        else:
            mask_valid = torch.ones_like(track_ids, dtype=torch.bool)
        if not torch.is_tensor(mask_valid):
            raise TypeError("batched mask_valid must be a torch tensor.")
        values = (*base_values, mask_valid)

        if labels.ndim != 2:
            raise ValueError("batched labels must have shape [B,Nmax].")
        if track_ids.ndim != 3 or track_ids.shape[2] != num_frames:
            raise ValueError("batched ids must have shape [B,Nmax,T].")
        if masks.ndim != 5 or masks.shape[2] != num_frames:
            raise ValueError("batched masks must have shape [B,Nmax,T,H,W].")
        if mask_valid.dtype != torch.bool or mask_valid.shape != track_ids.shape:
            raise ValueError("batched mask_valid must be bool with shape [B,Nmax,T].")
        pixel_valid = targets.get("pixel_valid")
        if pixel_valid is None:
            pixel_valid = torch.ones(
                (batch_size, num_frames, masks.shape[-2], masks.shape[-1]),
                dtype=torch.bool,
                device=track_ids.device,
            )
        if not torch.is_tensor(pixel_valid):
            raise TypeError("batched pixel_valid must be a torch tensor.")
        expected_pixel_valid_shape = (
            batch_size,
            num_frames,
            masks.shape[-2],
            masks.shape[-1],
        )
        if pixel_valid.dtype != torch.bool or tuple(pixel_valid.shape) != expected_pixel_valid_shape:
            raise ValueError(
                "batched pixel_valid must be bool with shape [B,T,H,W]."
            )
        values = (*values, pixel_valid)
        if track_counts.ndim != 1:
            raise ValueError("track_counts must have shape [B].")
        if (
            labels.shape[0] != batch_size
            or track_ids.shape[:2] != labels.shape
            or masks.shape[:2] != labels.shape
            or track_counts.shape[0] != batch_size
        ):
            raise ValueError("batched entity_targets must align with the image batch.")
        if min(masks.shape[-2:]) <= 0:
            raise ValueError("batched target masks must have non-empty spatial dimensions.")
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
        if labels.dtype not in integer_dtypes:
            raise TypeError("batched labels must use an integer dtype.")
        if track_ids.dtype not in integer_dtypes:
            raise TypeError("batched ids must use an integer dtype.")
        if track_counts.dtype not in integer_dtypes:
            raise TypeError("track_counts must use an integer dtype.")
        if any(value.device != device for value in values):
            raise ValueError("predictions and all batched target tensors must share one device.")

        max_tracks = labels.shape[1]
        clips: list[dict[str, torch.Tensor]] = []
        for batch_index in range(batch_size):
            count = int(track_counts[batch_index].item())
            if count < 0 or count > max_tracks:
                raise ValueError(
                    f"track_counts[{batch_index}]={count} is outside [0,{max_tracks}]."
                )
            if count < max_tracks:
                padded_ids = track_ids[batch_index, count:]
                padded_masks = masks[batch_index, count:].to(torch.float32)
                padded_labels = labels[batch_index, count:]
                padded_mask_valid = mask_valid[batch_index, count:]
                if padded_ids.numel() and not torch.all(padded_ids == -1):
                    raise ValueError("padded entity ID rows must be filled with -1.")
                if padded_masks.numel() and torch.any(padded_masks != 0):
                    raise ValueError("padded entity mask rows must be all zero.")
                if padded_labels.numel() and not torch.all(padded_labels == 0):
                    raise ValueError("padded entity label rows must be filled with 0.")
                if padded_mask_valid.numel() and padded_mask_valid.any():
                    raise ValueError("padded mask_valid rows must be false.")

            clips.append(
                {
                    "labels": labels[batch_index, :count],
                    "ids": track_ids[batch_index, :count],
                    "masks": masks[batch_index, :count],
                    "mask_valid": mask_valid[batch_index, :count],
                    "pixel_valid": pixel_valid[batch_index],
                }
            )
        return clips

    def _validate_targets(
        self,
        targets: Sequence[Mapping[str, Any]],
        *,
        batch_size: int,
        num_frames: int,
        num_queries: int,
        device: torch.device,
    ) -> tuple[list[dict[str, torch.Tensor]], bool]:
        if not isinstance(targets, Sequence) or len(targets) != batch_size:
            raise ValueError(f"targets must contain exactly {batch_size} clip dictionaries.")

        flattened: list[dict[str, torch.Tensor]] = []
        any_visible = False
        target_spatial_size: tuple[int, int] | None = None

        for clip_index, target in enumerate(targets):
            if not isinstance(target, Mapping):
                raise TypeError(f"targets[{clip_index}] must be a mapping.")
            missing = {"labels", "ids", "masks"}.difference(target)
            if missing:
                raise ValueError(
                    f"targets[{clip_index}] is missing keys: {sorted(missing)}"
                )

            labels = target["labels"]
            track_ids = target["ids"]
            masks = target["masks"]
            if not all(torch.is_tensor(value) for value in (labels, track_ids, masks)):
                raise TypeError("labels, ids, and masks must be torch tensors.")
            mask_valid = target.get(
                "mask_valid",
                torch.ones_like(track_ids, dtype=torch.bool),
            )
            if not torch.is_tensor(mask_valid):
                raise TypeError("mask_valid must be a torch tensor.")
            if mask_valid.dtype != torch.bool or mask_valid.shape != track_ids.shape:
                raise ValueError("clip mask_valid must be bool with shape [N,T].")

            if labels.ndim != 1:
                raise ValueError("clip labels must have shape [N].")
            if track_ids.ndim != 2 or track_ids.shape[1] != num_frames:
                raise ValueError(
                    f"clip ids must have shape [N,{num_frames}], "
                    f"got {tuple(track_ids.shape)}"
                )
            if masks.ndim != 4 or tuple(masks.shape[:2]) != (
                labels.shape[0],
                num_frames,
            ):
                raise ValueError(
                    "clip masks must have shape [N,T,H,W] and align with labels/ids."
                )
            if min(masks.shape[-2:]) <= 0:
                raise ValueError("target masks must have non-empty spatial dimensions.")
            pixel_valid = target.get(
                "pixel_valid",
                torch.ones(
                    (num_frames, masks.shape[-2], masks.shape[-1]),
                    dtype=torch.bool,
                    device=masks.device,
                ),
            )
            if not torch.is_tensor(pixel_valid):
                raise TypeError("pixel_valid must be a torch tensor.")
            if pixel_valid.dtype != torch.bool or tuple(pixel_valid.shape) != (
                num_frames,
                masks.shape[-2],
                masks.shape[-1],
            ):
                raise ValueError("pixel_valid must be bool with shape [T,H,W].")
            if track_ids.shape[0] != labels.shape[0]:
                raise ValueError("labels and ids must contain the same number of tracks.")

            spatial_size = (int(masks.shape[-2]), int(masks.shape[-1]))
            if target_spatial_size is None:
                target_spatial_size = spatial_size
            elif spatial_size != target_spatial_size:
                raise ValueError(
                    "all clips in a batch must use the same padded target-mask size."
                )

            integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
            if labels.dtype not in integer_dtypes:
                raise TypeError("labels must use an integer dtype.")
            if track_ids.dtype not in integer_dtypes:
                raise TypeError("ids must use an integer dtype.")
            if any(
                value.device != device
                for value in (labels, track_ids, masks, mask_valid, pixel_valid)
            ):
                raise ValueError("predictions and all target tensors must share one device.")
            if not torch.isfinite(masks).all():
                raise ValueError("target masks must contain only finite values.")
            if masks.numel() and ((masks < 0).any() or (masks > 1).any()):
                raise ValueError("target masks must contain values in [0, 1].")
            if labels.numel() and not torch.all(labels == 0):
                raise ValueError(
                    "VidEoMT adapter currently uses one foreground class; all labels must be 0."
                )
            if track_ids.numel() and (track_ids < -1).any():
                raise ValueError("track IDs must be non-negative or -1 for not visible.")

            track_ids = track_ids.to(dtype=torch.int64)
            mask_valid = mask_valid.to(dtype=torch.bool)
            pixel_valid = pixel_valid.to(dtype=torch.bool)
            masks = masks.to(dtype=torch.float32).masked_fill(
                ~pixel_valid.unsqueeze(0), 0.0
            )
            visible = track_ids >= 0
            if (visible & ~mask_valid).any():
                raise ValueError("unannotated frames must use track ID -1.")
            known_foreground = masks.flatten(2).sum(dim=2) > 0
            supervised_visible = visible & mask_valid & known_foreground
            track_count = int(supervised_visible.any(dim=1).sum().item())
            if track_count > num_queries:
                raise ValueError(
                    f"clip {clip_index} has {track_count} visible tracks but only "
                    f"{num_queries} entity slots."
                )
            if track_ids.shape[0] and (~visible.any(dim=1)).any():
                raise ValueError(
                    f"clip {clip_index} contains a track row that is never visible."
                )
            if track_ids.shape[0] and (~supervised_visible.any(dim=1)).any():
                raise ValueError(
                    f"clip {clip_index} contains a track with no pixel-valid visible mask "
                    "frame; refusing to create an ungrounded entity assignment."
                )

            seen_ids: set[int] = set()
            for row in range(track_ids.shape[0]):
                row_ids = track_ids[row, visible[row]]
                if row_ids.numel() == 0:
                    continue
                unique_ids = torch.unique(row_ids)
                if unique_ids.numel() != 1:
                    raise ValueError(
                        f"clip {clip_index} track row {row} changes ID across frames."
                    )
                track_id = int(unique_ids.item())
                if track_id in seen_ids:
                    raise ValueError(
                        f"clip {clip_index} contains duplicate track ID {track_id}."
                    )
                seen_ids.add(track_id)

                row_masks = masks[row]
                if (
                    row_masks[supervised_visible[row]].flatten(1).sum(dim=1) <= 0
                ).any():
                    raise ValueError(
                        f"clip {clip_index} has a pixel-valid visible track frame "
                        "with an empty mask."
                    )
                absent_masks = row_masks[~visible[row]].to(torch.float32)
                if (absent_masks.flatten(1).sum(dim=1) > 0).any():
                    raise ValueError(
                        f"clip {clip_index} must use zero masks wherever track ID is -1."
                    )

            any_visible = any_visible or track_count > 0
            labels = labels.to(dtype=torch.int64)
            for frame_index in range(num_frames):
                flattened.append(
                    {
                        "labels": labels,
                        "ids": track_ids[:, frame_index : frame_index + 1],
                        "masks": masks[:, frame_index : frame_index + 1],
                        "mask_valid": mask_valid[:, frame_index : frame_index + 1],
                        "pixel_valid": pixel_valid[frame_index : frame_index + 1],
                    }
                )

        return flattened, any_visible


    @staticmethod
    def _point_sample(
        masks: torch.Tensor,
        point_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Sample masks at normalized (x, y) coordinates using PointRend semantics."""
        grid = point_coords.mul(2.0).sub(1.0).unsqueeze(2)
        return F.grid_sample(
            masks,
            grid,
            mode="bilinear",
            align_corners=False,
        ).squeeze(-1)

    @staticmethod
    def _coords_from_indices(
        indices: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Convert flattened target-pixel indices to exact pixel-center coordinates."""
        x = torch.remainder(indices, width).to(torch.float32) + 0.5
        y = torch.div(indices, width, rounding_mode="floor").to(torch.float32) + 0.5
        return torch.stack((x / width, y / height), dim=-1)

    @torch.no_grad()
    def _match_with_pixel_valid(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Apply the pinned consistent matcher policy using known pixels only.

        Fully annotated masks continue through the upstream matcher unchanged. If
        either track-frame annotation or pixel coverage is partial, the first
        pixel-valid visible frame establishes each track's clip-consistent slot.
        Returned indices are CPU tensors, as in the upstream VidEoMT matcher.
        """
        matcher = self.matcher
        if all(
            bool(
                target["pixel_valid"].all().item()
                and target["mask_valid"].all().item()
            )
            for target in targets
        ):
            return matcher(outputs, targets)

        required = ("cost_class", "cost_mask", "cost_dice", "num_points", "frames")
        missing = [name for name in required if not hasattr(matcher, name)]
        if missing:
            raise TypeError(
                "Pixel-valid matching requires the pinned VidEoMT consistent matcher; "
                f"missing matcher attributes: {missing}."
            )
        if int(matcher.frames) != self.frames:
            raise ValueError("VidEoMT matcher frame count does not match the E3A adapter.")

        from scipy.optimize import linear_sum_assignment

        batch_frames, num_queries = outputs["pred_logits"].shape[:2]
        if batch_frames != len(targets) or batch_frames % self.frames:
            raise ValueError("Flattened outputs and targets must align in B*T order.")
        point_count = int(matcher.num_points)
        if point_count <= 0:
            raise ValueError("VidEoMT matcher num_points must be positive.")

        assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
        batch_size = batch_frames // self.frames
        for batch_index in range(batch_size):
            first_valid_frame: dict[int, int] = {}
            for frame_index in range(self.frames):
                flat_index = batch_index * self.frames + frame_index
                target = targets[flat_index]
                ids = target["ids"].reshape(-1)
                mask_valid = target["mask_valid"].reshape(-1)
                pixel_valid = target["pixel_valid"][0]
                masks = target["masks"][:, 0]
                known_foreground = (
                    (masks > 0) & pixel_valid.unsqueeze(0)
                ).flatten(1).any(dim=1)
                eligible = (ids >= 0) & mask_valid & known_foreground
                for row in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
                    first_valid_frame.setdefault(int(row), frame_index)

            tracks_by_first_frame: dict[int, list[int]] = {}
            for row, frame_index in first_valid_frame.items():
                tracks_by_first_frame.setdefault(frame_index, []).append(row)

            used_query_indices: list[int] = []
            matched_queries: list[int] = []
            matched_targets: list[int] = []
            for frame_index in sorted(tracks_by_first_frame):
                flat_index = batch_index * self.frames + frame_index
                target = targets[flat_index]
                used_targets = tracks_by_first_frame[frame_index]
                target_indices = torch.as_tensor(
                    used_targets,
                    dtype=torch.int64,
                    device=target["labels"].device,
                )
                probabilities = outputs["pred_logits"][flat_index].float().softmax(-1)
                target_classes = target["labels"][target_indices].to(torch.int64)
                cost_class = -probabilities[:, target_classes]

                valid_indices = torch.nonzero(
                    target["pixel_valid"][0].reshape(-1),
                    as_tuple=False,
                ).flatten()
                if valid_indices.numel() == 0:
                    raise RuntimeError(
                        "A pixel-valid entity match was selected without valid pixels."
                    )
                height, width = target["pixel_valid"].shape[-2:]
                sampled_indices = valid_indices[
                    torch.randint(
                        valid_indices.numel(),
                        (point_count,),
                        device=valid_indices.device,
                    )
                ]
                point_coords = self._coords_from_indices(
                    sampled_indices,
                    height=height,
                    width=width,
                )

                predicted_masks = outputs["pred_masks"][flat_index].float()
                target_masks = target["masks"][target_indices].to(
                    device=predicted_masks.device,
                    dtype=torch.float32,
                )
                predicted_points = self._point_sample(
                    predicted_masks,
                    point_coords.unsqueeze(0).expand(num_queries, -1, -1),
                ).squeeze(1)
                target_points = self._point_sample(
                    target_masks,
                    point_coords.unsqueeze(0).expand(len(used_targets), -1, -1),
                ).squeeze(1)

                positive_cost = F.binary_cross_entropy_with_logits(
                    predicted_points,
                    torch.ones_like(predicted_points),
                    reduction="none",
                )
                negative_cost = F.binary_cross_entropy_with_logits(
                    predicted_points,
                    torch.zeros_like(predicted_points),
                    reduction="none",
                )
                cost_mask = (
                    torch.einsum("kp,np->kn", positive_cost, target_points)
                    + torch.einsum("kp,np->kn", negative_cost, 1.0 - target_points)
                ) / float(point_count)

                predicted_probabilities = predicted_points.sigmoid()
                numerator = 2.0 * torch.einsum(
                    "kp,np->kn",
                    predicted_probabilities,
                    target_points,
                )
                denominator = (
                    predicted_probabilities.sum(dim=-1)[:, None]
                    + target_points.sum(dim=-1)[None, :]
                )
                cost_dice = 1.0 - (numerator + 1.0) / (denominator + 1.0)

                cost = (
                    float(matcher.cost_class) * cost_class
                    + float(matcher.cost_mask) * cost_mask
                    + float(matcher.cost_dice) * cost_dice
                )
                if not torch.isfinite(cost).all():
                    raise FloatingPointError("Pixel-valid Hungarian matching cost is non-finite.")
                if used_query_indices:
                    cost[used_query_indices, :] = 1e6

                query_rows, target_columns = linear_sum_assignment(
                    cost.detach().cpu().numpy()
                )
                used_query_indices.extend(int(value) for value in query_rows.tolist())
                matched_queries.extend(int(value) for value in query_rows.tolist())
                matched_targets.extend(
                    used_targets[int(column)] for column in target_columns.tolist()
                )

            for _ in range(self.frames):
                assignments.append(
                    (
                        torch.as_tensor(matched_queries, dtype=torch.int64),
                        torch.as_tensor(matched_targets, dtype=torch.int64),
                    )
                )
        return assignments

    def _loss_masks_with_pixel_valid(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        num_masks: float,
    ) -> dict[str, torch.Tensor]:
        """Compute VidEoMT BCE/Dice losses from valid pixels only."""
        logits_by_frame: list[torch.Tensor] = []
        labels_by_frame: list[torch.Tensor] = []
        candidate_count = max(
            int(self.num_points * self.oversample_ratio),
            self.num_points,
        )
        uncertain_count = min(
            int(self.num_points * self.importance_sample_ratio),
            self.num_points,
        )
        random_count = self.num_points - uncertain_count

        for frame_index, (source_indices, target_indices) in enumerate(indices):
            if target_indices.numel() == 0:
                continue
            target = targets[frame_index]
            valid_indices = torch.nonzero(
                target["pixel_valid"][0].reshape(-1),
                as_tuple=False,
            ).flatten()
            if valid_indices.numel() == 0:
                continue

            source_masks = outputs["pred_masks"][frame_index, source_indices].float()
            target_masks = target["masks"][target_indices].to(
                device=source_masks.device,
                dtype=torch.float32,
            )
            match_count = source_masks.shape[0]
            if match_count == 0:
                continue

            height, width = target["pixel_valid"].shape[-2:]
            candidate_positions = valid_indices[
                torch.randint(
                    valid_indices.numel(),
                    (match_count, candidate_count),
                    device=valid_indices.device,
                )
            ]
            candidate_coords = self._coords_from_indices(
                candidate_positions,
                height=height,
                width=width,
            )

            with torch.no_grad():
                candidate_logits = self._point_sample(
                    source_masks.detach(),
                    candidate_coords,
                ).squeeze(1)
                if uncertain_count:
                    uncertain_indices = torch.topk(
                        -candidate_logits.abs(),
                        k=uncertain_count,
                        dim=1,
                    ).indices
                    uncertain_coords = candidate_coords.gather(
                        1,
                        uncertain_indices.unsqueeze(-1).expand(-1, -1, 2),
                    )
                else:
                    uncertain_coords = candidate_coords[:, :0]

                if random_count:
                    random_positions = valid_indices[
                        torch.randint(
                            valid_indices.numel(),
                            (match_count, random_count),
                            device=valid_indices.device,
                        )
                    ]
                    random_coords = self._coords_from_indices(
                        random_positions,
                        height=height,
                        width=width,
                    )
                else:
                    random_coords = candidate_coords[:, :0]
                point_coords = torch.cat((uncertain_coords, random_coords), dim=1)

            logits_by_frame.append(
                self._point_sample(source_masks, point_coords).squeeze(1)
            )
            labels_by_frame.append(
                self._point_sample(target_masks, point_coords).squeeze(1)
            )

        if not logits_by_frame:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        point_logits = torch.cat(logits_by_frame, dim=0)
        point_labels = torch.cat(labels_by_frame, dim=0)
        normalizer = max(float(num_masks), 1.0)
        loss_mask = F.binary_cross_entropy_with_logits(
            point_logits,
            point_labels,
            reduction="none",
        ).mean(dim=1).sum() / normalizer

        probabilities = point_logits.sigmoid()
        numerator = 2.0 * (probabilities * point_labels).sum(dim=1)
        denominator = probabilities.sum(dim=1) + point_labels.sum(dim=1)
        loss_dice = (1.0 - (numerator + 1.0) / (denominator + 1.0)).sum() / normalizer
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}


    @staticmethod
    def _empty_unsupervised_target(
        target: Mapping[str, Any],
        *,
        clip_index: int,
    ) -> dict[str, torch.Tensor]:
        """Drop entity targets from a clip whose entity annotation is not valid."""
        labels = target.get("labels")
        track_ids = target.get("ids")
        masks = target.get("masks")
        if not all(torch.is_tensor(value) for value in (labels, track_ids, masks)):
            raise TypeError(
                f"Unsupervised clip {clip_index} must still provide tensor labels/ids/masks."
            )
        if (
            labels.ndim != 1
            or track_ids.ndim != 2
            or masks.ndim != 4
            or track_ids.shape[0] != labels.shape[0]
            or tuple(masks.shape[:2]) != tuple(track_ids.shape)
        ):
            raise ValueError(
                f"Unsupervised clip {clip_index} has malformed labels/ids/masks shapes."
            )
        mask_valid = target.get(
            "mask_valid",
            torch.ones_like(track_ids, dtype=torch.bool),
        )
        if (
            not torch.is_tensor(mask_valid)
            or mask_valid.dtype != torch.bool
            or tuple(mask_valid.shape) != tuple(track_ids.shape)
        ):
            raise ValueError(
                f"Unsupervised clip {clip_index} mask_valid must be bool [N,T]."
            )
        empty = {
            "labels": labels[:0],
            "ids": track_ids[:0],
            "masks": masks[:0],
            "mask_valid": mask_valid[:0],
        }
        if "pixel_valid" in target:
            empty["pixel_valid"] = target["pixel_valid"]
        return empty

    def _forward_impl(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch_or_targets: Mapping[str, Any] | Sequence[Mapping[str, torch.Tensor]],
        schedule_progress: float | None,
        *,
        return_matches: bool,
    ):
        """Shared implementation for loss-only and loss-plus-track-assignment calls."""
        import inspect

        if isinstance(batch_or_targets, Mapping):
            if "entity_targets" not in batch_or_targets:
                raise ValueError(
                    "VGGT Trainer batches must contain an 'entity_targets' target mapping."
                )
            raw_targets = batch_or_targets["entity_targets"]
            raw_entity_supervision_valid = batch_or_targets.get(
                "entity_supervision_valid"
            )
            raw_entity_frame_supervision_valid = batch_or_targets.get(
                "entity_frame_supervision_valid"
            )
            if raw_entity_frame_supervision_valid is None and isinstance(
                raw_targets, Mapping
            ):
                raw_entity_frame_supervision_valid = raw_targets.get(
                    "entity_frame_supervision_valid"
                )
            if raw_entity_supervision_valid is None and isinstance(raw_targets, Mapping):
                raw_entity_supervision_valid = raw_targets.get(
                    "entity_supervision_valid"
                )
        else:
            raw_targets = batch_or_targets
            raw_entity_supervision_valid = None
            raw_entity_frame_supervision_valid = None

        if schedule_progress is not None and not math.isfinite(float(schedule_progress)):
            raise ValueError("schedule_progress must be finite when supplied.")
        if "mask_logits" not in predictions or "objectness_logits" not in predictions:
            raise ValueError("predictions must contain mask_logits and objectness_logits.")

        mask_logits = predictions["mask_logits"]
        objectness_logits = predictions["objectness_logits"]
        if mask_logits.ndim != 5:
            raise ValueError("mask_logits must have shape [B,T,K,H,W].")
        if objectness_logits.ndim != 3:
            raise ValueError("objectness_logits must have shape [B,T,K].")
        batch_size, num_frames, num_queries, height, width = mask_logits.shape
        if num_frames != self.frames:
            raise ValueError(
                f"VidEoMT matcher is configured for {self.frames} frames, got {num_frames}."
            )
        if tuple(objectness_logits.shape) != (batch_size, num_frames, num_queries):
            raise ValueError("objectness_logits must align with mask_logits [B,T,K].")
        if batch_size <= 0 or num_queries <= 0 or height <= 0 or width <= 0:
            raise ValueError("prediction batch, entity, and mask dimensions must be non-empty.")
        if mask_logits.device != objectness_logits.device:
            raise ValueError("mask_logits and objectness_logits must share one device.")
        if not torch.isfinite(mask_logits).all() or not torch.isfinite(objectness_logits).all():
            raise ValueError("mask and objectness logits must be finite.")

        if isinstance(raw_targets, Mapping):
            targets = self._unpack_batched_targets(
                raw_targets,
                batch_size=batch_size,
                num_frames=num_frames,
                device=mask_logits.device,
            )
        else:
            if not isinstance(raw_targets, Sequence):
                raise TypeError("entity_targets must be a batch mapping or clip sequence.")
            targets = list(raw_targets)

        if raw_entity_supervision_valid is None:
            supervision_values = []
            for clip_index, target in enumerate(targets):
                if not isinstance(target, Mapping):
                    raise TypeError(f"entity_targets[{clip_index}] must be a mapping.")
                labels = target.get("labels")
                track_ids = target.get("ids")
                if not torch.is_tensor(labels) or not torch.is_tensor(track_ids):
                    raise TypeError(
                        f"entity_targets[{clip_index}] must contain tensor labels and ids."
                    )
                mask_valid = target.get(
                    "mask_valid",
                    torch.ones_like(track_ids, dtype=torch.bool),
                )
                if (
                    not torch.is_tensor(mask_valid)
                    or mask_valid.dtype != torch.bool
                    or tuple(mask_valid.shape) != tuple(track_ids.shape)
                ):
                    raise ValueError(
                        f"entity_targets[{clip_index}].mask_valid must be bool [N,T]."
                    )
                nested_valid = target.get("entity_supervision_valid")
                if nested_valid is not None and not isinstance(nested_valid, bool):
                    raise TypeError(
                        f"entity_targets[{clip_index}].entity_supervision_valid must be bool."
                    )
                supervision_values.append(
                    nested_valid
                    if nested_valid is not None
                    else bool(labels.numel() > 0 and mask_valid.any().item())
                )
            entity_supervision_valid = torch.tensor(
                supervision_values,
                dtype=torch.bool,
                device=mask_logits.device,
            )
        else:
            if (
                not torch.is_tensor(raw_entity_supervision_valid)
                or raw_entity_supervision_valid.dtype != torch.bool
                or tuple(raw_entity_supervision_valid.shape) != (batch_size,)
            ):
                raise ValueError(
                    "entity_supervision_valid must be bool with shape [B]."
                )
            entity_supervision_valid = raw_entity_supervision_valid.to(
                device=mask_logits.device
            )
        if raw_entity_frame_supervision_valid is None:
            frame_valid_values = []
            for clip_index, target in enumerate(targets):
                frame_valid = target.get("entity_frame_supervision_valid")
                if frame_valid is None:
                    labels = target.get("labels")
                    if not torch.is_tensor(labels):
                        raise TypeError(
                            f"entity_targets[{clip_index}] must contain tensor labels."
                        )
                    # Do not infer complete object coverage from image/pixel validity.
                    # The sole implicit positive case is a source-verified empty clip.
                    verified_empty = (
                        labels.numel() == 0
                        and bool(entity_supervision_valid[clip_index].item())
                    )
                    frame_valid = torch.full(
                        (num_frames,),
                        verified_empty,
                        dtype=torch.bool,
                        device=mask_logits.device,
                    )
                if (
                    not torch.is_tensor(frame_valid)
                    or frame_valid.dtype != torch.bool
                    or tuple(frame_valid.shape) != (num_frames,)
                ):
                    raise ValueError(
                        f"entity_targets[{clip_index}].entity_frame_supervision_valid "
                        "must be bool [T]."
                    )
                frame_valid_values.append(frame_valid)
            entity_frame_supervision_valid = torch.stack(frame_valid_values, dim=0)
        else:
            if (
                not torch.is_tensor(raw_entity_frame_supervision_valid)
                or raw_entity_frame_supervision_valid.dtype != torch.bool
                or tuple(raw_entity_frame_supervision_valid.shape) != (
                    batch_size,
                    num_frames,
                )
            ):
                raise ValueError(
                    "entity_frame_supervision_valid must be bool with shape [B,T]."
                )
            entity_frame_supervision_valid = raw_entity_frame_supervision_valid

        entity_frame_supervision_valid = entity_frame_supervision_valid.to(
            device=mask_logits.device
        ) & entity_supervision_valid[:, None]

        targets = [
            target
            if is_valid
            else self._empty_unsupervised_target(target, clip_index=clip_index)
            for clip_index, (target, is_valid) in enumerate(
                zip(targets, entity_supervision_valid.tolist(), strict=True)
            )
        ]

        flattened_targets, any_visible = self._validate_targets(
            targets,
            batch_size=batch_size,
            num_frames=num_frames,
            num_queries=num_queries,
            device=mask_logits.device,
        )

        flat_masks = mask_logits.float().reshape(
            batch_size * num_frames,
            num_queries,
            1,
            height,
            width,
        )
        # Class logits are clip-consistent, but unknown frames must not enter
        # the temporal pooling that produces those logits.
        pixel_frame_valid = torch.tensor(
            [
                bool(target["pixel_valid"].all().item())
                for target in flattened_targets
            ],
            dtype=torch.bool,
            device=objectness_logits.device,
        ).reshape(batch_size, num_frames)
        class_frame_valid = (
            entity_frame_supervision_valid & pixel_frame_valid
        )
        class_frame_weights = class_frame_valid.unsqueeze(-1).to(
            dtype=torch.float32
        )
        clip_objectness = (
            (objectness_logits.float() * class_frame_weights).sum(
                dim=1, keepdim=True
            )
            / class_frame_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        flat_objectness = clip_objectness.expand(-1, num_frames, -1)
        no_object_logits = torch.zeros_like(flat_objectness)
        flat_logits = torch.stack((flat_objectness, no_object_logits), dim=-1).reshape(
            batch_size * num_frames, num_queries, 2
        )
        upstream_outputs = {"pred_logits": flat_logits, "pred_masks": flat_masks}


        supervised_batch_indices = (
            entity_supervision_valid.nonzero(as_tuple=False).flatten().tolist()
        )
        supervised_flat_indices = [
            batch_index * num_frames + frame_index
            for batch_index in supervised_batch_indices
            for frame_index in range(num_frames)
        ]

        if not supervised_flat_indices:
            zero_ce = flat_logits.sum() * 0.0
            zero_mask = flat_masks.sum() * 0.0
            losses = {
                "loss_ce": zero_ce,
                "loss_mask": zero_mask,
                "loss_dice": zero_mask,
            }
            supervised_indices = None
        else:
            flat_index_tensor = torch.as_tensor(
                supervised_flat_indices,
                dtype=torch.int64,
                device=flat_masks.device,
            )
            supervised_outputs = {
                "pred_logits": flat_logits.index_select(0, flat_index_tensor),
                "pred_masks": flat_masks.index_select(0, flat_index_tensor),
            }
            supervised_targets = [
                flattened_targets[index] for index in supervised_flat_indices
            ]
            all_masks_annotated = all(
                bool(
                    target["mask_valid"].all().item()
                    and target["pixel_valid"].all().item()
                )
                for target in supervised_targets
            )
            supervised_indices = None
            if any_visible and all_masks_annotated and bool(class_frame_valid.all().item()):
                if return_matches:
                    criterion_forward = self.criterion.forward
                    supports_match_return = (
                        "ret_match_result" in inspect.signature(criterion_forward).parameters
                    )
                    if supports_match_return:
                        losses, supervised_indices = self.criterion(
                            supervised_outputs,
                            supervised_targets,
                            ret_match_result=True,
                        )
                    else:
                        supervised_indices = self.matcher(
                            supervised_outputs,
                            supervised_targets,
                        )
                        losses = self.criterion(
                            supervised_outputs,
                            supervised_targets,
                        )
                else:
                    losses = self.criterion(supervised_outputs, supervised_targets)
            else:
                matcher = self.matcher
                if matcher is None or not hasattr(self.criterion, "loss_labels"):
                    raise TypeError(
                        "Partially annotated or all-empty batches require the official "
                        "matcher and VideoSetCriterion loss methods."
                    )
                supervised_indices = self._match_with_pixel_valid(
                    supervised_outputs,
                    supervised_targets,
                )
                losses = {}
                criterion_losses = getattr(
                    self.criterion,
                    "losses",
                    ("labels", "masks"),
                )
                if "labels" in criterion_losses:
                    # VidEoMT's class loss labels every unmatched query as no-object.
                    # A frame containing any unknown pixels cannot safely provide
                    # those negatives, even though its known pixels may be used by
                    # the mask matcher/loss.
                    class_frame_valid_flat = class_frame_valid.reshape(-1)
                    class_valid_indices = [
                        index
                        for index, flat_index in enumerate(supervised_flat_indices)
                        if bool(class_frame_valid_flat[flat_index].item())
                    ]
                    if class_valid_indices:
                        class_index_tensor = torch.as_tensor(
                            class_valid_indices,
                            dtype=torch.int64,
                            device=supervised_outputs["pred_logits"].device,
                        )
                        class_outputs = {
                            "pred_logits": supervised_outputs["pred_logits"].index_select(
                                0, class_index_tensor
                            )
                        }
                        class_targets = [
                            supervised_targets[index] for index in class_valid_indices
                        ]
                        class_matches = [
                            supervised_indices[index] for index in class_valid_indices
                        ]
                        losses.update(
                            self.criterion.loss_labels(
                                class_outputs,
                                class_targets,
                                class_matches,
                                num_masks=1.0,
                            )
                        )
                    else:
                        losses["loss_ce"] = (
                            supervised_outputs["pred_logits"].sum() * 0.0
                        )
                if "masks" in criterion_losses:
                    zero_mask_loss = supervised_outputs["pred_masks"].sum() * 0.0
                    if any_visible:
                        valid_indices = []
                        valid_match_count = 0
                        for target, (source_index, target_index) in zip(
                            supervised_targets,
                            supervised_indices,
                            strict=True,
                        ):
                            valid_for_matches = target["mask_valid"].reshape(-1)[
                                target_index
                            ]
                            pixel_has_valid = target["pixel_valid"].any()
                            target_ids = target["ids"].reshape(-1)[target_index]
                            target_has_foreground = (
                                target["masks"][target_index].flatten(1).sum(dim=1) > 0
                            )
                            valid_for_matches = valid_for_matches & pixel_has_valid & (
                                (target_ids < 0) | target_has_foreground
                            )
                            valid_for_matches = valid_for_matches.to(
                                device=source_index.device
                            )
                            valid_indices.append(
                                (
                                    source_index[valid_for_matches],
                                    target_index[valid_for_matches],
                                )
                            )
                            valid_match_count += int(valid_for_matches.sum().item())
                        if valid_match_count:
                            has_unknown_pixels = any(
                                not bool(target["pixel_valid"].all().item())
                                for target in supervised_targets
                            )
                            if has_unknown_pixels:
                                losses.update(
                                    self._loss_masks_with_pixel_valid(
                                        supervised_outputs,
                                        supervised_targets,
                                        valid_indices,
                                        num_masks=float(valid_match_count),
                                    )
                                )
                            else:
                                if not hasattr(self.criterion, "loss_masks"):
                                    raise TypeError(
                                        "Partially annotated clips require "
                                        "VideoSetCriterion.loss_masks."
                                    )
                                losses.update(
                                    self.criterion.loss_masks(
                                        supervised_outputs,
                                        supervised_targets,
                                        valid_indices,
                                        num_masks=float(valid_match_count),
                                    )
                                )
                        else:
                            losses.update(
                                {
                                    "loss_mask": zero_mask_loss,
                                    "loss_dice": zero_mask_loss,
                                }
                            )
                    else:
                        losses.update(
                            {
                                "loss_mask": zero_mask_loss,
                                "loss_dice": zero_mask_loss,
                            }
                        )
                unsupported = set(criterion_losses).difference({"labels", "masks"})
                if unsupported:
                    raise ValueError(
                        "E3A adapter only supports VidEoMT labels/masks losses, "
                        f"got {sorted(unsupported)}."
                    )

        if return_matches:
            empty_pair = (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
            indices = [empty_pair for _ in range(batch_size * num_frames)]
            if supervised_flat_indices:
                if supervised_indices is None:
                    raise RuntimeError(
                        "VidEoMT did not return matches for supervised entity clips."
                    )
                for local_index, flat_index in enumerate(supervised_flat_indices):
                    indices[flat_index] = supervised_indices[local_index]
        else:
            indices = supervised_indices

        missing_losses = set(self._LOSS_KEYS).difference(losses)
        if missing_losses:
            raise RuntimeError(
                f"VidEoMT criterion did not return required losses: {sorted(missing_losses)}"
            )
        objective = sum(
            losses[key] * weight for key, weight in self.loss_weights.items()
        )
        result = {**losses, "loss_objective": objective}

        if not return_matches:
            return result

        if indices is None or len(indices) != batch_size * num_frames:
            raise RuntimeError("VidEoMT did not return one match result per clip frame.")
        track_to_slot: list[dict[int, int]] = [dict() for _ in range(batch_size)]
        for batch_index in range(batch_size):
            mapping = track_to_slot[batch_index]
            for frame_index in range(num_frames):
                flat_index = batch_index * num_frames + frame_index
                source_indices, target_indices = indices[flat_index]
                frame_track_ids = flattened_targets[flat_index]["ids"].reshape(-1)
                for source_index, target_index in zip(
                    source_indices.tolist(), target_indices.tolist(), strict=True
                ):
                    track_id = int(frame_track_ids[target_index].item())
                    if track_id < 0:
                        continue
                    previous_slot = mapping.get(track_id)
                    if previous_slot is not None and previous_slot != source_index:
                        raise RuntimeError(
                            "VidEoMT returned inconsistent query slots for clip track "
                            f"{track_id}: {previous_slot} then {source_index}."
                        )
                    mapping[track_id] = int(source_index)
        return result, track_to_slot

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch_or_targets: Mapping[str, Any] | Sequence[Mapping[str, torch.Tensor]],
        schedule_progress: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return VidEoMT's mask/class losses in the VGGT Trainer contract."""
        return self._forward_impl(
            predictions,
            batch_or_targets,
            schedule_progress,
            return_matches=False,
        )

    def forward_with_matches(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch_or_targets: Mapping[str, Any] | Sequence[Mapping[str, torch.Tensor]],
        schedule_progress: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[int, int]]]:
        """Return mask losses plus GT-track-ID to predicted-slot assignments."""
        return self._forward_impl(
            predictions,
            batch_or_targets,
            schedule_progress,
            return_matches=True,
        )
