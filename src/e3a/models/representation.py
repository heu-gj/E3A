"""Composable E3A representation model for masks and 3D entity states."""

from __future__ import annotations

import torch
from torch import nn

from e3a.geometry.entity_pooling import pool_entity_geometry
from e3a.geometry.vggt_geometry import decode_and_unproject
from e3a.models.encoder import E3AEncoder
from e3a.models.heads.entity_mask_head import EntityMaskHead


class E3ARepresentationModel(nn.Module):
    """Compose VGGT-Ω entity tokens, SAM 3 mask prediction, and 3D pooling.

    This module intentionally stops before future action anticipation. The
    target/action/time heads and their losses depend on the audited dataset
    event schema and are not guessed here.
    """

    def __init__(
        self,
        encoder: E3AEncoder,
        mask_head: EntityMaskHead,
        *,
        min_entity_mass: float = 1.0,
    ) -> None:
        super().__init__()
        if min_entity_mass < 0:
            raise ValueError("min_entity_mass must be non-negative.")
        self.encoder = encoder
        self.mask_head = mask_head
        self.min_entity_mass = float(min_entity_mass)

    def forward(
        self,
        images: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(images, timestamps)
        required = {"entity_tokens", "fused_features", "pose_enc", "depth"}
        missing = required.difference(outputs)
        if missing:
            raise RuntimeError(
                "E3A 3D entity representation is missing required VGGT-Ω "
                f"features: {sorted(missing)}"
            )
        if outputs["pose_enc"] is None or outputs["depth"] is None:
            raise RuntimeError(
                "E3A 3D entity representation requires enabled VGGT-Ω "
                "CameraHead and DenseHead outputs."
            )

        mask_outputs = self.mask_head(
            outputs["entity_tokens"],
            outputs["fused_features"],
        )
        if "mask_logits" not in mask_outputs:
            raise RuntimeError("EntityMaskHead must return mask_logits.")
        outputs.update(mask_outputs)

        geometry = decode_and_unproject(
            outputs["depth"],
            outputs["pose_enc"],
        )
        outputs.update(geometry)
        outputs.update(
            pool_entity_geometry(
                outputs["mask_logits"],
                geometry["points_world"],
                outputs["depth"],
                min_mass=self.min_entity_mass,
            )
        )
        return outputs
