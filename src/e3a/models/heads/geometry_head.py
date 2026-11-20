"""E3A geometry head delegating camera/depth math to official VGGT-Ω code."""

from __future__ import annotations

import torch
from torch import nn

from e3a.geometry.entity_pooling import pool_entity_geometry
from e3a.geometry.vggt_geometry import decode_and_unproject


class GeometryHead(nn.Module):
    """Decode VGGT-Ω geometry and pool mask-conditioned 3D entity states."""

    def __init__(self, *, min_entity_mass: float = 1.0) -> None:
        super().__init__()
        if min_entity_mass < 0:
            raise ValueError("min_entity_mass must be non-negative.")
        self.min_entity_mass = float(min_entity_mass)

    def forward(
        self,
        *,
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        geometry = decode_and_unproject(depth, pose_enc)
        pooled = pool_entity_geometry(
            mask_logits,
            geometry["points_world"],
            depth,
            min_mass=self.min_entity_mass,
        )
        return {**geometry, **pooled}
