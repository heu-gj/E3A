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

        extrinsics = geometry["extrinsics_camera_from_world"]
        rotation = extrinsics[..., :3]
        translation = extrinsics[..., 3]
        centers_camera = (
            torch.matmul(
                rotation.unsqueeze(2),
                pooled["entity_centers_world"].unsqueeze(-1),
            ).squeeze(-1)
            + translation.unsqueeze(2)
        )
        covariance_camera = (
            rotation.unsqueeze(2)
            @ pooled["entity_covariance_world"]
            @ rotation.transpose(-1, -2).unsqueeze(2)
        )
        geometry_valid = pooled["entity_geometry_valid"]
        centers_camera = torch.where(
            geometry_valid.unsqueeze(-1),
            centers_camera,
            torch.zeros_like(centers_camera),
        )
        covariance_camera = torch.where(
            geometry_valid[..., None, None],
            covariance_camera,
            torch.zeros_like(covariance_camera),
        )

        return {
            **geometry,
            **pooled,
            "entity_centers_camera": centers_camera,
            "entity_covariance_camera": covariance_camera,
        }
