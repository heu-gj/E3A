"""Batch collation helpers for E3A's clip-level entity supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def collate_entity_targets(
    clip_targets: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    """Pad variable-length per-clip tracks into batch-first tensors.

    Each input clip has:
        labels: [N]
        ids: [N,T], using -1 for frames without a visible annotation
        masks: [N,T,H,W]

    Returns a mapping suitable for VGGT-Ω's recursive gradient-accumulation
    chunker:
        labels: [B,Nmax]
        ids: [B,Nmax,T], padded with -1
        masks: [B,Nmax,T,H,W], padded with zero
        track_counts: [B]

    All clips must already use the same frame count and padded mask resolution.
    """
    if not isinstance(clip_targets, Sequence) or len(clip_targets) == 0:
        raise ValueError("clip_targets must be a non-empty sequence.")

    normalized: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    batch_size = len(clip_targets)
    max_tracks = 0
    num_frames: int | None = None
    mask_size: tuple[int, int] | None = None
    device: torch.device | None = None

    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)

    for clip_index, target in enumerate(clip_targets):
        if not isinstance(target, Mapping):
            raise TypeError(f"clip_targets[{clip_index}] must be a mapping.")
        missing = {"labels", "ids", "masks"}.difference(target)
        if missing:
            raise ValueError(
                f"clip_targets[{clip_index}] is missing keys: {sorted(missing)}"
            )

        labels, track_ids, masks = target["labels"], target["ids"], target["masks"]
        if not all(torch.is_tensor(value) for value in (labels, track_ids, masks)):
            raise TypeError("labels, ids, and masks must be torch tensors.")
        if labels.ndim != 1:
            raise ValueError("clip labels must have shape [N].")
        if track_ids.ndim != 2:
            raise ValueError("clip ids must have shape [N,T].")
        if masks.ndim != 4:
            raise ValueError("clip masks must have shape [N,T,H,W].")
        if track_ids.shape[0] != labels.shape[0]:
            raise ValueError("labels and ids must contain the same number of tracks.")
        if tuple(masks.shape[:2]) != tuple(track_ids.shape):
            raise ValueError("masks and ids must have matching [N,T] dimensions.")
        if min(masks.shape[-2:]) <= 0:
            raise ValueError("clip masks must have non-empty spatial dimensions.")
        if labels.dtype not in integer_dtypes or track_ids.dtype not in integer_dtypes:
            raise TypeError("labels and ids must use integer dtypes.")

        clip_device = labels.device
        if track_ids.device != clip_device or masks.device != clip_device:
            raise ValueError("labels, ids, and masks in each clip must share one device.")
        if device is None:
            device = clip_device
        elif clip_device != device:
            raise ValueError("all clips in a batch must share one device.")

        clip_frames = int(track_ids.shape[1])
        clip_mask_size = (int(masks.shape[-2]), int(masks.shape[-1]))
        if num_frames is None:
            num_frames = clip_frames
        elif clip_frames != num_frames:
            raise ValueError("all clips in a batch must use the same frame count.")
        if mask_size is None:
            mask_size = clip_mask_size
        elif clip_mask_size != mask_size:
            raise ValueError("all clips in a batch must use the same padded mask size.")

        masks_float = masks.to(torch.float32)
        if not torch.isfinite(masks_float).all():
            raise ValueError(f"clip_targets[{clip_index}] masks must be finite.")
        if masks_float.numel() and (
            (masks_float < 0).any() or (masks_float > 1).any()
        ):
            raise ValueError("entity masks must contain values in [0,1].")

        normalized.append(
            (
                labels.to(torch.int64),
                track_ids.to(torch.int64),
                masks_float,
            )
        )
        max_tracks = max(max_tracks, int(labels.shape[0]))

    assert device is not None and num_frames is not None and mask_size is not None
    height, width = mask_size
    labels_batch = torch.zeros(
        batch_size,
        max_tracks,
        dtype=torch.int64,
        device=device,
    )
    ids_batch = torch.full(
        (batch_size, max_tracks, num_frames),
        -1,
        dtype=torch.int64,
        device=device,
    )
    masks_batch = torch.zeros(
        batch_size,
        max_tracks,
        num_frames,
        height,
        width,
        dtype=torch.float32,
        device=device,
    )
    track_counts = torch.zeros(batch_size, dtype=torch.int64, device=device)

    for batch_index, (labels, track_ids, masks) in enumerate(normalized):
        count = labels.shape[0]
        track_counts[batch_index] = count
        if count == 0:
            continue
        labels_batch[batch_index, :count] = labels
        ids_batch[batch_index, :count] = track_ids
        masks_batch[batch_index, :count] = masks

    return {
        "labels": labels_batch,
        "ids": ids_batch,
        "masks": masks_batch,
        "track_counts": track_counts,
    }



def collate_e3a_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate canonical HOI4D/Ego-Exo4D clip samples for E3A joint training.

    Per-sample optional targets:
      hand_targets: keypoints_camera [T,2,21,3], valid [T,2,21]
      interaction_targets: contact/valid [T,2,N]
      anticipation_targets: target_entity_track_id, target_entity_valid,
                            action_label, action_valid, time_to_event, time_valid

    Missing task annotations are represented by zero tensors plus false
    validity masks. They are never converted into negative supervision.
    """
    if not isinstance(samples, Sequence) or len(samples) == 0:
        raise ValueError("samples must be a non-empty sequence.")

    images = [sample.get("images") for sample in samples]
    timestamps = [sample.get("timestamps") for sample in samples]
    entity_targets = [sample.get("entity_targets") for sample in samples]
    if not all(torch.is_tensor(value) for value in images):
        raise TypeError("Every sample must contain an image tensor.")
    if not all(torch.is_tensor(value) for value in timestamps):
        raise TypeError("Every sample must contain a timestamp tensor.")
    if not all(isinstance(value, Mapping) for value in entity_targets):
        raise TypeError("Every sample must contain an entity_targets mapping.")

    image_shape = tuple(images[0].shape)
    if len(image_shape) != 4 or image_shape[1] != 3:
        raise ValueError("Sample images must have shape [T,3,H,W].")
    frame_count = image_shape[0]
    for index, (image, time) in enumerate(zip(images, timestamps, strict=True)):
        if tuple(image.shape) != image_shape:
            raise ValueError("All sample images must share [T,3,H,W] dimensions.")
        if time.shape != (frame_count,):
            raise ValueError(f"samples[{index}] timestamps must have shape [T].")
        if not torch.isfinite(time).all():
            raise ValueError("timestamps must be finite.")
        if frame_count > 1 and (time[1:] < time[:-1]).any():
            raise ValueError("timestamps must be monotonically non-decreasing.")

    entity_batch = collate_entity_targets(entity_targets)
    batch_size = len(samples)
    device = images[0].device
    hand_keypoints = torch.zeros(
        batch_size, frame_count, 2, 21, 3, dtype=torch.float32, device=device
    )
    hand_valid = torch.zeros(
        batch_size, frame_count, 2, 21, dtype=torch.bool, device=device
    )
    hand_visibility = torch.zeros_like(hand_valid, dtype=torch.float32)
    hand_visibility_valid = torch.zeros_like(hand_valid)

    max_tracks = entity_batch["labels"].shape[1]
    contact = torch.zeros(
        batch_size, frame_count, 2, max_tracks, dtype=torch.float32, device=device
    )
    contact_valid = torch.zeros_like(contact, dtype=torch.bool)

    target_entity_ids = torch.full(
        (batch_size,), -1, dtype=torch.int64, device=device
    )
    target_entity_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    action_labels = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
    action_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    time_to_event = torch.zeros(batch_size, dtype=torch.float32, device=device)
    time_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for batch_index, sample in enumerate(samples):
        hand = sample.get("hand_targets")
        if hand is not None:
            if not isinstance(hand, Mapping):
                raise TypeError("hand_targets must be a mapping.")
            points = hand["keypoints_camera"]
            valid = hand["valid"]
            if tuple(points.shape) != (frame_count, 2, 21, 3):
                raise ValueError("hand keypoints must have shape [T,2,21,3].")
            if tuple(valid.shape) != (frame_count, 2, 21):
                raise ValueError("hand valid mask must have shape [T,2,21].")
            hand_keypoints[batch_index] = points.to(device=device, dtype=torch.float32)
            hand_valid[batch_index] = valid.to(device=device, dtype=torch.bool)
            if "visibility" in hand:
                visibility = hand["visibility"]
                visibility_valid = hand["visibility_valid"]
                if tuple(visibility.shape) != (frame_count, 2, 21):
                    raise ValueError("hand visibility must have shape [T,2,21].")
                if tuple(visibility_valid.shape) != (frame_count, 2, 21):
                    raise ValueError("hand visibility_valid must have shape [T,2,21].")
                hand_visibility[batch_index] = visibility.to(
                    device=device, dtype=torch.float32
                )
                hand_visibility_valid[batch_index] = visibility_valid.to(
                    device=device, dtype=torch.bool
                )

        relation = sample.get("interaction_targets")
        if relation is not None:
            if not isinstance(relation, Mapping):
                raise TypeError("interaction_targets must be a mapping.")
            relation_contact = relation["contact"]
            relation_valid = relation["valid"]
            track_count = int(entity_batch["track_counts"][batch_index].item())
            expected = (frame_count, 2, track_count)
            if tuple(relation_contact.shape) != expected or tuple(relation_valid.shape) != expected:
                raise ValueError(
                    f"interaction targets for sample {batch_index} must have shape {expected}."
                )
            if track_count:
                contact[batch_index, :, :, :track_count] = relation_contact.to(
                    device=device, dtype=torch.float32
                )
                contact_valid[batch_index, :, :, :track_count] = relation_valid.to(
                    device=device, dtype=torch.bool
                )

        anticipation = sample.get("anticipation_targets")
        if anticipation is not None:
            if not isinstance(anticipation, Mapping):
                raise TypeError("anticipation_targets must be a mapping.")
            target_entity_ids[batch_index] = int(
                anticipation.get("target_entity_track_id", -1)
            )
            target_entity_valid[batch_index] = bool(
                anticipation.get("target_entity_valid", False)
            )
            action_labels[batch_index] = int(anticipation.get("action_label", -1))
            action_valid[batch_index] = bool(anticipation.get("action_valid", False))
            time_to_event[batch_index] = float(
                anticipation.get("time_to_event", 0.0)
            )
            time_valid[batch_index] = bool(anticipation.get("time_valid", False))

    hand_targets_batch = {
        "keypoints_camera": hand_keypoints,
        "valid": hand_valid,
        "visibility": hand_visibility,
        "visibility_valid": hand_visibility_valid,
    }
    return {
        "images": torch.stack(images, dim=0),
        "timestamps": torch.stack(
            [value.to(device=device, dtype=torch.float32) for value in timestamps],
            dim=0,
        ),
        "entity_targets": entity_batch,
        "hand_targets": hand_targets_batch,
        "interaction_targets": {
            "contact": contact,
            "valid": contact_valid,
        },
        "anticipation_targets": {
            "target_entity_track_ids": target_entity_ids,
            "target_entity_valid": target_entity_valid,
            "action_labels": action_labels,
            "action_valid": action_valid,
            "time_to_event": time_to_event,
            "time_valid": time_valid,
        },
        "dataset": [str(sample.get("dataset", "unknown")) for sample in samples],
        "seq_name": [str(sample.get("seq_name", "unknown")) for sample in samples],
    }
