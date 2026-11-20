"""Thin E3A adapter around VidEoMT's official video matcher and criterion."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
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
        self.loss_weights = normalized_weights
        self.criterion = criterion
        self._matcher_fallback = matcher if not hasattr(criterion, "matcher") else None

    @staticmethod
    def _load_upstream_classes():
        repo_root = Path(__file__).resolve().parents[3]
        upstream_root = repo_root / "third_party" / "videomt"
        if not upstream_root.is_dir():
            raise ImportError(
                "VidEoMT source is missing. Run "
                "git submodule update --init third_party/videomt."
            )
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))

        try:
            from videomt.criterion_videomt import VideoSetCriterion
            from videomt.modeling.matcher import VideoHungarianMatcher_Consistent
        except ImportError as exc:
            raise ImportError(
                "VidEoMT and its optional dependencies are required. Initialize "
                "third_party/videomt and install compatible SciPy and Detectron2 "
                "dependencies in the target PyTorch/CUDA environment."
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
        """Unpack padded tensor targets that remain batch-aligned under upstream chunking."""
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
        if not all(
            torch.is_tensor(value)
            for value in (labels, track_ids, masks, track_counts)
        ):
            raise TypeError("batched entity_targets values must all be torch tensors.")
        if labels.ndim != 2:
            raise ValueError("batched labels must have shape [B,Nmax].")
        if track_ids.ndim != 3 or track_ids.shape[2] != num_frames:
            raise ValueError("batched ids must have shape [B,Nmax,T].")
        if masks.ndim != 5 or masks.shape[2] != num_frames:
            raise ValueError("batched masks must have shape [B,Nmax,T,H,W].")
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
        if any(
            value.device != device
            for value in (labels, track_ids, masks, track_counts)
        ):
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
                if padded_ids.numel() and not torch.all(padded_ids == -1):
                    raise ValueError("padded entity ID rows must be filled with -1.")
                if padded_masks.numel() and torch.any(padded_masks != 0):
                    raise ValueError("padded entity mask rows must be all zero.")
                if padded_labels.numel() and not torch.all(padded_labels == 0):
                    raise ValueError("padded entity label rows must be filled with 0.")

            clips.append(
                {
                    "labels": labels[batch_index, :count],
                    "ids": track_ids[batch_index, :count],
                    "masks": masks[batch_index, :count],
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
            if track_ids.shape[0] != labels.shape[0]:
                raise ValueError("labels and ids must contain the same number of tracks.")

            spatial_size = (int(masks.shape[-2]), int(masks.shape[-1]))
            if target_spatial_size is None:
                target_spatial_size = spatial_size
            elif spatial_size != target_spatial_size:
                raise ValueError(
                    "all clips in a batch must use the same padded target-mask size."
                )

            if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("labels must use an integer dtype.")
            if track_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError("ids must use an integer dtype.")
            if labels.device != device or track_ids.device != device or masks.device != device:
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

            visible = track_ids >= 0
            track_count = int(visible.any(dim=1).sum().item())
            if track_count > num_queries:
                raise ValueError(
                    f"clip {clip_index} has {track_count} visible tracks but only "
                    f"{num_queries} entity slots."
                )
            if track_ids.shape[0] and (~visible.any(dim=1)).any():
                raise ValueError(
                    f"clip {clip_index} contains a track row that is never visible."
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
                if (row_masks[visible[row]].flatten(1).sum(dim=1) <= 0).any():
                    raise ValueError(
                        f"clip {clip_index} has a visible track frame with an empty mask."
                    )
                missing_masks = row_masks[~visible[row]].to(torch.float32)
                if (missing_masks.flatten(1).sum(dim=1) > 0).any():
                    raise ValueError(
                        f"clip {clip_index} must use zero masks wherever track ID is -1."
                    )

            any_visible = any_visible or track_count > 0
            labels = labels.to(dtype=torch.int64)
            track_ids = track_ids.to(dtype=torch.int64)
            masks = masks.to(dtype=torch.float32)
            for frame_index in range(num_frames):
                flattened.append(
                    {
                        "labels": labels,
                        "ids": track_ids[:, frame_index : frame_index + 1],
                        "masks": masks[:, frame_index : frame_index + 1],
                    }
                )

        return flattened, any_visible

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch_or_targets: Mapping[str, Any] | Sequence[Mapping[str, torch.Tensor]],
        schedule_progress: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Accept either a clip-target list or the official VGGT Trainer call.

        VGGT Trainer calls every criterion as loss(predictions, batch,
        schedule_progress). In that form, the clip target list must be stored
        under batch["entity_targets"]. In training, that value is a padded tensor
        mapping with labels, ids, masks, and track_counts; a per-clip target list
        is also accepted for direct use and unit tests.
        """
        if isinstance(batch_or_targets, Mapping):
            if "entity_targets" not in batch_or_targets:
                raise ValueError(
                    "VGGT Trainer batches must contain an 'entity_targets' target mapping."
                )
            raw_targets = batch_or_targets["entity_targets"]
        else:
            raw_targets = batch_or_targets

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
            targets = raw_targets

        flattened_targets, any_visible = self._validate_targets(
            targets,
            batch_size=batch_size,
            num_frames=num_frames,
            num_queries=num_queries,
            device=mask_logits.device,
        )

        # VidEoMT expects one mask frame per flattened B*T entry. The
        # consistent matcher enforces one query/track assignment per clip.
        flat_masks = mask_logits.float().reshape(
            batch_size * num_frames,
            num_queries,
            1,
            height,
            width,
        )
        clip_objectness = objectness_logits.float().mean(dim=1, keepdim=True)
        flat_objectness = clip_objectness.expand(-1, num_frames, -1)
        no_object_logits = torch.zeros_like(flat_objectness)
        flat_logits = torch.stack(
            (flat_objectness, no_object_logits),
            dim=-1,
        ).reshape(batch_size * num_frames, num_queries, 2)

        upstream_outputs = {
            "pred_logits": flat_logits,
            "pred_masks": flat_masks,
        }

        if any_visible:
            losses = self.criterion(upstream_outputs, flattened_targets)
        else:
            matcher = self.matcher
            if matcher is None or not hasattr(self.criterion, "loss_labels"):
                raise TypeError(
                    "All-empty batches require the official matcher and loss_labels method."
                )
            indices = matcher(upstream_outputs, flattened_targets)
            losses = self.criterion.loss_labels(
                upstream_outputs,
                flattened_targets,
                indices,
                num_masks=1.0,
            )
            zero_mask_loss = flat_masks.sum() * 0.0
            losses = {
                **losses,
                "loss_mask": zero_mask_loss,
                "loss_dice": zero_mask_loss,
            }

        missing_losses = set(self._LOSS_KEYS).difference(losses)
        if missing_losses:
            raise RuntimeError(
                f"VidEoMT criterion did not return required losses: {sorted(missing_losses)}"
            )
        objective = sum(
            losses[key] * weight for key, weight in self.loss_weights.items()
        )
        return {**losses, "loss_objective": objective}
