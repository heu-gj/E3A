"""Joint multi-task objective for the unified E3A model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class E3AMultiTaskLoss(nn.Module):
    """Combine entity masks, geometry, hand pose, interaction, and anticipation losses.

    The mask criterion supplies clip-consistent GT-track-to-query assignments.
    Those same assignments are reused to supervise geometry, interaction
    pairs, and the future target slot, avoiding a second tracking model.
    """

    _WEIGHT_NAMES = (
        "entity",
        "geometry",
        "hand",
        "hand_visibility",
        "interaction",
        "target_entity",
        "action",
        "time",
    )

    def __init__(
        self,
        mask_criterion: nn.Module,
        *,
        weights: Mapping[str, float] | None = None,
        hand_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(mask_criterion, nn.Module):
            raise TypeError("mask_criterion must be a torch.nn.Module.")
        normalized = {name: 1.0 for name in self._WEIGHT_NAMES}
        if weights is not None:
            unknown = set(weights).difference(normalized)
            if unknown:
                raise ValueError(f"Unknown E3A loss weights: {sorted(unknown)}")
            normalized.update({key: float(value) for key, value in weights.items()})
        if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
            raise ValueError("E3A loss weights must be finite and non-negative.")
        if not any(normalized.values()):
            raise ValueError("At least one E3A loss weight must be positive.")
        if not math.isfinite(hand_beta) or hand_beta <= 0:
            raise ValueError("hand_beta must be finite and positive.")
        self.mask_criterion = mask_criterion
        self.weights = normalized
        self.hand_beta = float(hand_beta)

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.float().sum() * 0.0

    @staticmethod
    def _require_tensor(
        mapping: Mapping[str, Any],
        key: str,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if key not in mapping or not torch.is_tensor(mapping[key]):
            raise ValueError(f"Expected tensor target '{key}'.")
        value = mapping[key]
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError(
                f"Target '{key}' must have shape {shape}, got {tuple(value.shape)}."
            )
        return value

    def _geometry_losses(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        track_to_slot: list[dict[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = self._zero(predictions["mask_logits"])
        if "geometry_targets" not in batch:
            return zero, zero, zero

        targets = batch["geometry_targets"]
        entity_targets = batch.get("entity_targets")
        if not isinstance(targets, Mapping) or not isinstance(entity_targets, Mapping):
            raise TypeError("geometry_targets and entity_targets must be mappings.")

        predicted_centers = self._require_tensor(predictions, "entity_centers_camera")
        predicted_covariance = self._require_tensor(
            predictions, "entity_covariance_camera"
        )
        predicted_valid = self._require_tensor(predictions, "entity_geometry_valid")
        target_centers = self._require_tensor(targets, "centers_camera")
        target_covariance = self._require_tensor(targets, "covariance_camera")
        target_valid = self._require_tensor(targets, "valid")
        ids = self._require_tensor(entity_targets, "ids")
        counts = self._require_tensor(entity_targets, "track_counts")

        if predicted_centers.ndim != 4 or predicted_centers.shape[-1] != 3:
            raise ValueError("Predicted entity centers must have shape [B,T,K,3].")
        batch_size, frames, slots, _ = predicted_centers.shape
        if predicted_covariance.shape != (batch_size, frames, slots, 3, 3):
            raise ValueError("Predicted entity covariance must have shape [B,T,K,3,3].")
        if predicted_valid.shape != (batch_size, frames, slots) or predicted_valid.dtype != torch.bool:
            raise TypeError("entity_geometry_valid must be bool [B,T,K].")
        if target_centers.ndim != 4 or target_centers.shape[0] != batch_size:
            raise ValueError("Target centers must have shape [B,N,T,3].")
        max_tracks = target_centers.shape[1]
        if target_centers.shape != (batch_size, max_tracks, frames, 3):
            raise ValueError("Target centers must have shape [B,N,T,3].")
        if target_covariance.shape != (batch_size, max_tracks, frames, 3, 3):
            raise ValueError("Target covariance must have shape [B,N,T,3,3].")
        if target_valid.shape != (batch_size, max_tracks, frames) or target_valid.dtype != torch.bool:
            raise TypeError("geometry_targets.valid must be bool [B,N,T].")
        if ids.shape != (batch_size, max_tracks, frames):
            raise ValueError("entity target IDs must have shape [B,N,T].")
        if counts.shape != (batch_size,):
            raise ValueError("entity track_counts must have shape [B].")
        if len(track_to_slot) != batch_size:
            raise ValueError("Track-slot assignments must align with geometry batch.")
        if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("entity target IDs must use an integer dtype.")
        if counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("entity track_counts must use an integer dtype.")

        predicted_centers = predicted_centers.float()
        predicted_covariance = predicted_covariance.float()
        target_centers = target_centers.to(
            device=predicted_centers.device, dtype=torch.float32
        )
        target_covariance = target_covariance.to(
            device=predicted_centers.device, dtype=torch.float32
        )
        target_valid = target_valid.to(device=predicted_centers.device)
        predicted_valid = predicted_valid.to(device=predicted_centers.device)
        ids = ids.to(device=predicted_centers.device, dtype=torch.int64)
        counts = counts.to(device=predicted_centers.device, dtype=torch.int64)

        clip_center_losses: list[torch.Tensor] = []
        clip_covariance_losses: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            count = int(counts[batch_index].item())
            if count < 0 or count > max_tracks:
                raise ValueError("entity track count is outside the padded target range.")
            matched_predictions: list[torch.Tensor] = []
            matched_targets: list[torch.Tensor] = []
            matched_predicted_covariance: list[torch.Tensor] = []
            matched_target_covariance: list[torch.Tensor] = []

            for track_index in range(count):
                track_ids = ids[batch_index, track_index]
                visible_ids = track_ids[track_ids >= 0]
                if visible_ids.numel() == 0:
                    continue
                unique_ids = torch.unique(visible_ids)
                if unique_ids.numel() != 1:
                    raise ValueError(
                        "Each HOI4D entity target row must retain one clip-local track ID."
                    )
                track_id = int(unique_ids[0].item())
                slot = track_to_slot[batch_index].get(track_id)
                if slot is None:
                    continue
                if slot < 0 or slot >= slots:
                    raise RuntimeError("VidEoMT returned an entity slot outside model output.")

                frame_valid = (
                    target_valid[batch_index, track_index]
                    & (track_ids >= 0)
                    & predicted_valid[batch_index, :, slot]
                )
                if not frame_valid.any():
                    continue
                frame_indices = frame_valid.nonzero(as_tuple=False).flatten()
                selected_target_centers = target_centers[
                    batch_index, track_index, frame_indices
                ]
                selected_predicted_centers = predicted_centers[
                    batch_index, frame_indices, slot
                ]
                selected_target_covariance = target_covariance[
                    batch_index, track_index, frame_indices
                ]
                selected_predicted_covariance = predicted_covariance[
                    batch_index, frame_indices, slot
                ]
                if not torch.isfinite(selected_target_centers).all() or not torch.isfinite(
                    selected_target_covariance
                ).all():
                    raise ValueError("Valid HOI4D geometry targets must be finite.")
                if not torch.isfinite(selected_predicted_centers).all() or not torch.isfinite(
                    selected_predicted_covariance
                ).all():
                    raise FloatingPointError("Valid predicted entity geometry must be finite.")

                matched_targets.append(selected_target_centers)
                matched_predictions.append(selected_predicted_centers)
                matched_target_covariance.append(selected_target_covariance)
                matched_predicted_covariance.append(selected_predicted_covariance)

            if not matched_targets:
                continue

            target_center_values = torch.cat(matched_targets, dim=0)
            predicted_center_values = torch.cat(matched_predictions, dim=0)
            target_covariance_values = torch.cat(matched_target_covariance, dim=0)
            predicted_covariance_values = torch.cat(matched_predicted_covariance, dim=0)

            # Depth units remain source-native. Separate per-clip scales preserve
            # relative 3D layout and shape without pretending raw values are meters.
            target_scale = target_center_values[:, 2].abs().median().clamp_min(1e-6)
            predicted_scale = predicted_center_values[:, 2].abs().median().clamp_min(1e-6)
            center_loss = F.smooth_l1_loss(
                predicted_center_values / predicted_scale,
                target_center_values / target_scale,
                reduction="mean",
            )
            covariance_loss = F.smooth_l1_loss(
                predicted_covariance_values / predicted_scale.square(),
                target_covariance_values / target_scale.square(),
                reduction="mean",
            )
            clip_center_losses.append(center_loss)
            clip_covariance_losses.append(covariance_loss)

        if not clip_center_losses:
            return zero, zero, zero
        loss_center = torch.stack(clip_center_losses).mean()
        loss_covariance = torch.stack(clip_covariance_losses).mean()
        return loss_center, loss_covariance, loss_center + loss_covariance

    def _hand_losses(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self._zero(predictions["hand_keypoints_camera"])
        if "hand_targets" not in batch:
            return zero, zero
        targets = batch["hand_targets"]
        if not isinstance(targets, Mapping):
            raise TypeError("hand_targets must be a mapping.")
        predicted = predictions["hand_keypoints_camera"]
        target = self._require_tensor(
            targets,
            "keypoints_camera",
            shape=tuple(predicted.shape),
        ).to(device=predicted.device, dtype=torch.float32)
        valid = self._require_tensor(
            targets,
            "valid",
            shape=tuple(predicted.shape[:-1]),
        )
        if valid.dtype != torch.bool:
            raise TypeError("hand_targets.valid must use torch.bool.")
        valid = valid.to(device=predicted.device)

        if valid.any():
            selected_target = target[valid]
            selected_prediction = predicted.float()[valid]
            if not torch.isfinite(selected_target).all():
                raise ValueError("Valid 3D hand keypoints must be finite.")
            if not torch.isfinite(selected_prediction).all():
                raise FloatingPointError("Predicted 3D hand keypoints must be finite.")
            hand_loss = F.smooth_l1_loss(
                selected_prediction,
                selected_target,
                beta=self.hand_beta,
                reduction="sum",
            ) / (valid.sum().float() * 3.0).clamp_min(1.0)
        else:
            hand_loss = zero

        visibility_loss = zero
        if "visibility" in targets:
            visibility = self._require_tensor(
                targets,
                "visibility",
                shape=tuple(predictions["hand_joint_visibility_logits"].shape),
            ).to(device=predicted.device, dtype=torch.float32)
            visibility_valid = self._require_tensor(
                targets,
                "visibility_valid",
                shape=tuple(visibility.shape),
            )
            if visibility_valid.dtype != torch.bool:
                raise TypeError("hand_targets.visibility_valid must use torch.bool.")
            visibility_valid = visibility_valid.to(device=predicted.device)
            if visibility_valid.any():
                selected_visibility = visibility[visibility_valid]
                selected_visibility_logits = predictions[
                    "hand_joint_visibility_logits"
                ].float()[visibility_valid]
                if not torch.isfinite(selected_visibility).all() or (
                    (selected_visibility < 0).any() or (selected_visibility > 1).any()
                ):
                    raise ValueError("Valid hand visibility targets must be finite in [0,1].")
                if not torch.isfinite(selected_visibility_logits).all():
                    raise FloatingPointError("Predicted hand visibility logits must be finite.")
                visibility_loss = F.binary_cross_entropy_with_logits(
                    selected_visibility_logits,
                    selected_visibility,
                    reduction="mean",
                )
        return hand_loss, visibility_loss

    def _interaction_loss(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        track_to_slot: list[dict[int, int]],
    ) -> torch.Tensor:
        logits = predictions["interaction_logits"]
        zero = self._zero(logits)
        if "interaction_targets" not in batch:
            return zero
        if "entity_targets" not in batch:
            raise ValueError("interaction_targets require entity_targets for GT track mapping.")

        interaction_targets = batch["interaction_targets"]
        entity_targets = batch["entity_targets"]
        if not isinstance(interaction_targets, Mapping) or not isinstance(entity_targets, Mapping):
            raise TypeError("interaction_targets and entity_targets must be mappings.")
        contact = self._require_tensor(interaction_targets, "contact")
        valid = self._require_tensor(interaction_targets, "valid")
        if valid.dtype != torch.bool:
            raise TypeError("interaction_targets.valid must use torch.bool.")
        ids = self._require_tensor(entity_targets, "ids")
        counts = self._require_tensor(entity_targets, "track_counts")

        batch_size, frames, hands, slots = logits.shape
        if contact.ndim != 4 or tuple(contact.shape[:3]) != (batch_size, frames, hands):
            raise ValueError("interaction contact targets must have shape [B,T,H,Nmax].")
        if valid.shape != contact.shape or ids.ndim != 3:
            raise ValueError("interaction valid and entity IDs must have compatible shapes.")
        if ids.shape[0] != batch_size or ids.shape[2] != frames:
            raise ValueError("entity target IDs must have shape [B,Nmax,T].")
        if contact.shape[3] != ids.shape[1] or counts.shape != (batch_size,):
            raise ValueError("interaction targets must align with padded entity tracks.")
        if not torch.isfinite(contact).all() or (
            contact.numel() and ((contact < 0).any() or (contact > 1).any())
        ):
            raise ValueError("contact targets must be finite values in [0,1].")

        mapped_targets = torch.zeros_like(logits, dtype=torch.float32)
        mapped_valid = torch.zeros_like(logits, dtype=torch.bool)
        ids = ids.to(device=logits.device, dtype=torch.int64)
        contact = contact.to(device=logits.device, dtype=torch.float32)
        valid = valid.to(device=logits.device, dtype=torch.bool)
        counts = counts.to(device=logits.device, dtype=torch.int64)

        for batch_index in range(batch_size):
            count = int(counts[batch_index].item())
            for track_index in range(count):
                visible_ids = ids[batch_index, track_index]
                visible_ids = visible_ids[visible_ids >= 0]
                if visible_ids.numel() == 0:
                    continue
                track_id = int(visible_ids[0].item())
                slot = track_to_slot[batch_index].get(track_id)
                if slot is None:
                    continue
                if slot >= slots:
                    raise RuntimeError("VidEoMT returned an entity slot outside model output.")
                mapped_targets[batch_index, :, :, slot] = contact[
                    batch_index, :, :, track_index
                ]
                mapped_valid[batch_index, :, :, slot] = valid[
                    batch_index, :, :, track_index
                ]

        if not mapped_valid.any():
            return zero
        raw = F.binary_cross_entropy_with_logits(
            logits.float(),
            mapped_targets,
            reduction="none",
        )
        return (raw * mapped_valid.float()).sum() / mapped_valid.sum().float().clamp_min(1.0)

    def _anticipation_losses(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        track_to_slot: list[dict[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_logits = predictions["target_entity_logits"]
        action_logits = predictions["action_logits"]
        time_mu = predictions["time_log_mean"]
        time_log_scale = predictions["time_log_scale"]
        zero_target = self._zero(target_logits)
        zero_action = self._zero(action_logits)
        zero_time = self._zero(time_mu)

        if "anticipation_targets" not in batch:
            return zero_target, zero_action, zero_time
        targets = batch["anticipation_targets"]
        if not isinstance(targets, Mapping):
            raise TypeError("anticipation_targets must be a mapping.")
        batch_size, classes = target_logits.shape
        if len(track_to_slot) != batch_size:
            raise ValueError("Track-slot assignments must align with anticipation batch.")

        target_valid = self._require_tensor(
            targets, "target_entity_valid", shape=(batch_size,)
        )
        if target_valid.dtype != torch.bool:
            raise TypeError("anticipation_targets.target_entity_valid must use torch.bool.")
        target_valid = target_valid.to(device=target_logits.device)
        target_ids = self._require_tensor(
            targets, "target_entity_track_ids", shape=(batch_size,)
        )
        if target_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("anticipation_targets.target_entity_track_ids must be integer.")
        target_ids = target_ids.to(device=target_logits.device, dtype=torch.int64)
        if target_valid.any():
            selected_target_ids = target_ids[target_valid]
            if (selected_target_ids < -1).any():
                raise ValueError("Valid target entity IDs must be -1 (no target) or non-negative.")
            if not torch.isfinite(target_logits[target_valid]).all():
                raise FloatingPointError("Valid target-entity logits must be finite.")
            labels = torch.zeros(batch_size, dtype=torch.int64, device=target_logits.device)
            for batch_index in target_valid.nonzero(as_tuple=False).flatten().tolist():
                track_id = int(target_ids[batch_index].item())
                if track_id < 0:
                    # Class zero means explicitly annotated "no target entity".
                    labels[batch_index] = 0
                else:
                    slot = track_to_slot[batch_index].get(track_id)
                    if slot is None:
                        raise ValueError(
                            f"Future target track {track_id} is not visible/matched in "
                            f"observed clip {batch_index}; mark target_entity_valid=False "
                            "unless the dataset defines an explicit unseen-target class."
                        )
                    if slot + 1 >= classes:
                        raise RuntimeError("Matched target slot exceeds target-logit classes.")
                    labels[batch_index] = slot + 1
            target_loss = F.cross_entropy(target_logits[target_valid], labels[target_valid])
        else:
            target_loss = zero_target

        action_valid = self._require_tensor(
            targets, "action_valid", shape=(batch_size,)
        )
        if action_valid.dtype != torch.bool:
            raise TypeError("anticipation_targets.action_valid must use torch.bool.")
        action_valid = action_valid.to(device=action_logits.device)
        action_labels = self._require_tensor(
            targets, "action_labels", shape=(batch_size,)
        )
        if action_labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("anticipation_targets.action_labels must be integer.")
        action_labels = action_labels.to(device=action_logits.device, dtype=torch.int64)
        if action_valid.any():
            selected = action_labels[action_valid]
            if not torch.isfinite(action_logits[action_valid]).all():
                raise FloatingPointError("Valid action logits must be finite.")
            if (selected < 0).any() or (selected >= action_logits.shape[-1]).any():
                raise ValueError("Valid action labels must index the configured action vocabulary.")
            action_loss = F.cross_entropy(action_logits[action_valid], selected)
        else:
            action_loss = zero_action

        time_valid = self._require_tensor(
            targets, "time_valid", shape=(batch_size,)
        )
        if time_valid.dtype != torch.bool:
            raise TypeError("anticipation_targets.time_valid must use torch.bool.")
        time_valid = time_valid.to(device=time_mu.device)
        time_values = self._require_tensor(
            targets, "time_to_event", shape=(batch_size,)
        ).to(device=time_mu.device, dtype=torch.float32)
        if time_valid.any():
            if not torch.isfinite(time_mu[time_valid]).all() or not torch.isfinite(
                time_log_scale[time_valid]
            ).all():
                raise FloatingPointError(
                    "Valid time-to-event distribution parameters must be finite."
                )
            selected_time = time_values[time_valid]
            if not torch.isfinite(selected_time).all() or (selected_time <= 0).any():
                raise ValueError("Valid time_to_event labels must be finite positive seconds.")
            log_time = selected_time.log()
            scale = time_log_scale[time_valid].float().exp().clamp_min(1e-4)
            normalized = (log_time - time_mu[time_valid].float()) / scale
            # Negative log-likelihood of a log-normal event-time distribution.
            time_loss = (
                0.5 * normalized.square()
                + time_log_scale[time_valid].float()
                + log_time
                + 0.5 * math.log(2.0 * math.pi)
            ).mean()
        else:
            time_loss = zero_time
        return target_loss, action_loss, time_loss

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        schedule_progress: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("E3A training batch must be a mapping.")
        if not hasattr(self.mask_criterion, "forward_with_matches"):
            raise TypeError("mask_criterion must expose forward_with_matches().")

        entity_losses, track_to_slot = self.mask_criterion.forward_with_matches(
            predictions,
            batch,
            schedule_progress,
        )
        loss_entity = entity_losses["loss_objective"]
        loss_geometry_center, loss_geometry_covariance, loss_geometry = (
            self._geometry_losses(predictions, batch, track_to_slot)
        )
        loss_hand, loss_hand_visibility = self._hand_losses(predictions, batch)
        loss_interaction = self._interaction_loss(predictions, batch, track_to_slot)
        loss_target, loss_action, loss_time = self._anticipation_losses(
            predictions,
            batch,
            track_to_slot,
        )

        objective = (
            self.weights["entity"] * loss_entity
            + self.weights["geometry"] * loss_geometry
            + self.weights["hand"] * loss_hand
            + self.weights["hand_visibility"] * loss_hand_visibility
            + self.weights["interaction"] * loss_interaction
            + self.weights["target_entity"] * loss_target
            + self.weights["action"] * loss_action
            + self.weights["time"] * loss_time
        )
        return {
            "loss_entity": loss_entity,
            "loss_geometry_center": loss_geometry_center,
            "loss_geometry_covariance": loss_geometry_covariance,
            "loss_geometry": loss_geometry,
            "loss_hand": loss_hand,
            "loss_hand_visibility": loss_hand_visibility,
            "loss_interaction": loss_interaction,
            "loss_target_entity": loss_target,
            "loss_action": loss_action,
            "loss_time": loss_time,
            "loss_objective": objective,
        }
