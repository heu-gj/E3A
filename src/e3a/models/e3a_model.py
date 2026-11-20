"""Unified E3A model: one VGGT-Ω encoder and parallel task heads."""

from __future__ import annotations

import torch
from torch import nn

from e3a.models.encoder import E3AEncoder
from e3a.models.entity_state import PersistentEntityPool
from e3a.models.heads.anticipation_head import AnticipationHead
from e3a.models.heads.entity_mask_head import EntityMaskHead
from e3a.models.heads.geometry_head import GeometryHead
from e3a.models.heads.hand_head import HandPoseHead
from e3a.models.heads.interaction_head import InteractionHead


def camera_points_to_world(
    points_camera: torch.Tensor,
    extrinsics_camera_from_world: torch.Tensor,
) -> torch.Tensor:
    """Transform camera-frame points using VGGT-Ω's camera-from-world pose."""
    if points_camera.ndim != 5 or points_camera.shape[-1] != 3:
        raise ValueError("points_camera must have shape [B,T,H,J,3].")
    if extrinsics_camera_from_world.shape != (
        points_camera.shape[0],
        points_camera.shape[1],
        3,
        4,
    ):
        raise ValueError("camera extrinsics must have shape [B,T,3,4].")

    rotation = extrinsics_camera_from_world[..., :3]
    translation = extrinsics_camera_from_world[..., 3]
    centered = points_camera - translation[:, :, None, None, :]
    return torch.einsum("btji,bthkj->bthki", rotation, centered)


def _infer_entity_feature_dim(encoder: nn.Module) -> int | None:
    """Infer the post-Aggregator token width, including VGGT-Ω's two streams."""
    backbone = getattr(encoder, "backbone", None)
    model = getattr(backbone, "model", None)
    aggregator = getattr(model, "aggregator", None)
    camera_token = getattr(aggregator, "camera_token", None)
    if torch.is_tensor(camera_token) and camera_token.ndim >= 1:
        return int(camera_token.shape[-1]) * 2
    token_bank = getattr(encoder, "token_bank", None)
    token_dim = getattr(token_bank, "embed_dim", None)
    return int(token_dim) if token_dim is not None else None


class E3AModel(nn.Module):
    """One-forward multi-task E3A model.

    VGGT-Ω is the only online visual backbone. Entity slots, hand registers,
    and the future register are injected by E3AEncoder and decoded by parallel
    lightweight task heads.
    """

    def __init__(
        self,
        encoder: E3AEncoder,
        *,
        num_actions: int,
        entity_dim: int | None = None,
        pixel_dim: int = 256,
        mask_dim: int = 256,
        interaction_dim: int = 256,
        min_entity_mass: float = 1.0,
        mask_head: nn.Module | None = None,
        geometry_head: nn.Module | None = None,
        hand_head: nn.Module | None = None,
        interaction_head: nn.Module | None = None,
        anticipation_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_actions <= 0:
            raise ValueError(
                "num_actions must be positive and come from the audited action vocabulary."
            )
        entity_dim = _infer_entity_feature_dim(encoder) if entity_dim is None else entity_dim
        if entity_dim is None or entity_dim <= 0:
            raise ValueError(
                "entity_dim must be provided or inferable from the VGGT-Ω Aggregator."
            )

        self.encoder = encoder
        self.mask_head = mask_head or EntityMaskHead(
            entity_dim=entity_dim,
            pixel_dim=pixel_dim,
            mask_dim=mask_dim,
        )
        self.geometry_head = geometry_head or GeometryHead(
            min_entity_mass=min_entity_mass
        )
        self.entity_pool = PersistentEntityPool(entity_dim)
        self.hand_head = hand_head or HandPoseHead(embed_dim=entity_dim)
        self.interaction_head = interaction_head or InteractionHead(
            embed_dim=entity_dim,
            interaction_dim=interaction_dim,
        )
        self.anticipation_head = anticipation_head or AnticipationHead(
            embed_dim=entity_dim,
            interaction_dim=interaction_dim,
            num_actions=num_actions,
        )

    @staticmethod
    def _validate_encoder_output(outputs: dict[str, torch.Tensor]) -> None:
        required = {
            "entity_tokens",
            "hand_tokens",
            "future_tokens",
            "fused_features",
            "depth",
            "pose_enc",
        }
        missing = required.difference(outputs)
        if missing:
            raise RuntimeError(
                f"VGGT-Ω/E3A encoder output is missing {sorted(missing)}"
            )
        if outputs["pose_enc"] is None or outputs["depth"] is None:
            raise RuntimeError(
                "E3A requires the official VGGT-Ω CameraHead and DenseHead outputs."
            )

    def forward(
        self,
        images: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if timestamps.ndim == 1:
            timestamps = timestamps.unsqueeze(0)

        outputs = self.encoder(images, timestamps)
        self._validate_encoder_output(outputs)

        mask_outputs = self.mask_head(
            outputs["entity_tokens"],
            outputs["fused_features"],
        )
        geometry_outputs = self.geometry_head(
            depth=outputs["depth"],
            pose_enc=outputs["pose_enc"],
            mask_logits=mask_outputs["mask_logits"],
        )
        entity_state = self.entity_pool(
            outputs["entity_tokens"],
            entity_centers_world=geometry_outputs["entity_centers_world"],
            entity_geometry_valid=geometry_outputs["entity_geometry_valid"],
        )
        hand_outputs = self.hand_head(outputs["hand_tokens"])
        hand_keypoints_world = camera_points_to_world(
            hand_outputs["hand_keypoints_camera"],
            geometry_outputs["extrinsics_camera_from_world"],
        )
        interaction_outputs = self.interaction_head(
            entity_tokens=outputs["entity_tokens"],
            hand_tokens=outputs["hand_tokens"],
            hand_keypoints_world=hand_keypoints_world,
            entity_centers_world=geometry_outputs["entity_centers_world"],
            entity_geometry_valid=geometry_outputs["entity_geometry_valid"],
            timestamps=timestamps,
        )
        anticipation_outputs = self.anticipation_head(
            future_tokens=outputs["future_tokens"],
            persistent_entity_tokens=entity_state["persistent_entity_tokens"],
            interaction_tokens=interaction_outputs["interaction_tokens"],
            interaction_logits=interaction_outputs["interaction_logits"],
            objectness_logits=mask_outputs["objectness_logits"],
        )

        return {
            **outputs,
            **mask_outputs,
            **geometry_outputs,
            **entity_state,
            **hand_outputs,
            "hand_keypoints_world": hand_keypoints_world,
            **interaction_outputs,
            **anticipation_outputs,
        }
