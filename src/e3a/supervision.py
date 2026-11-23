"""Shared shape validation for annotation-coverage flags."""

from __future__ import annotations

from typing import Any

import torch


def frame_flags(
    value: Any, *, batch: int, frames: int, device: torch.device, name: str,
    per_clip: bool = False, default: bool = False,
) -> torch.Tensor:
    """Normalize explicit booleans to [B,T], without moving target devices.

    Missing coverage is false. Per-clip [T] flags and batched [B]/[B,1]/[B,T]
    flags are distinguished explicitly, including when B happens to equal T.
    """
    if value is None or isinstance(value, bool):
        return torch.full((batch, frames), default if value is None else value,
                          dtype=torch.bool, device=device)
    if not torch.is_tensor(value) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be boolean.")
    if value.device != device:
        raise ValueError(f"{name} must share the prediction/target device.")
    if value.ndim == 0:
        return value.expand(batch, frames)
    if per_clip and batch == 1 and value.shape == (frames,):
        return value[None]
    if value.shape == (batch,):
        return value[:, None].expand(batch, frames)
    if value.shape == (batch, 1):
        return value.expand(batch, frames)
    if value.shape == (batch, frames):
        return value
    raise ValueError(f"{name} must be scalar, [B], [B,1] or [B,T] (clip: [T]).")
