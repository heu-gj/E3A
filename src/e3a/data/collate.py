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
