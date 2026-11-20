"""Geometry-conditioned hand-object relation head."""

from __future__ import annotations

import torch
from torch import nn


class InteractionHead(nn.Module):
    """Encode pairwise hand/entity state with a small shared MLP."""

    def __init__(
        self,
        *,
        embed_dim: int = 2048,
        interaction_dim: int = 256,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(embed_dim, interaction_dim, feature_dim) <= 0:
            raise ValueError("Interaction head dimensions must be positive.")
        self.hand_projection = nn.Linear(embed_dim, feature_dim)
        self.entity_projection = nn.Linear(embed_dim, feature_dim)
        pair_dim = feature_dim * 2 + 8
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, interaction_dim),
            nn.GELU(),
            nn.Linear(interaction_dim, interaction_dim),
            nn.GELU(),
        )
        self.contact_head = nn.Linear(interaction_dim, 1)

    @staticmethod
    def _relative_velocity(
        relative_position: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        velocity = torch.zeros_like(relative_position)
        if relative_position.shape[1] <= 1:
            return velocity
        dt = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(1e-3)
        velocity[:, 1:] = (
            relative_position[:, 1:] - relative_position[:, :-1]
        ) / dt[:, :, None, None, None]
        return velocity

    def forward(
        self,
        *,
        entity_tokens: torch.Tensor,
        hand_tokens: torch.Tensor,
        hand_keypoints_world: torch.Tensor,
        entity_centers_world: torch.Tensor,
        entity_geometry_valid: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if entity_tokens.ndim != 4 or hand_tokens.ndim != 4:
            raise ValueError("Entity and hand tokens must be [B,T,N,D].")
        batch, frames, slots, _ = entity_tokens.shape
        if hand_tokens.shape[:2] != (batch, frames):
            raise ValueError("Hand and entity tokens must align in [B,T].")
        hands = hand_tokens.shape[2]
        if hand_keypoints_world.shape != (batch, frames, hands, 21, 3):
            raise ValueError("hand_keypoints_world must have shape [B,T,H,21,3].")
        if entity_centers_world.shape != (batch, frames, slots, 3):
            raise ValueError("entity_centers_world must have shape [B,T,K,3].")
        if entity_geometry_valid.shape != (batch, frames, slots):
            raise ValueError("entity_geometry_valid must have shape [B,T,K].")
        if timestamps.shape != (batch, frames):
            raise ValueError("timestamps must have shape [B,T].")

        wrist = hand_keypoints_world[:, :, :, 0, :]
        relative = wrist.unsqueeze(3) - entity_centers_world.unsqueeze(2)
        velocity = self._relative_velocity(relative, timestamps)
        distance = relative.norm(dim=-1, keepdim=True)
        valid = entity_geometry_valid[:, :, None, :, None].to(relative.dtype)

        hand_tokens = hand_tokens.to(dtype=self.hand_projection.weight.dtype)
        entity_tokens = entity_tokens.to(dtype=self.entity_projection.weight.dtype)
        hand_features = self.hand_projection(hand_tokens).unsqueeze(3).expand(
            -1, -1, -1, slots, -1
        )
        entity_features = self.entity_projection(entity_tokens).unsqueeze(2).expand(
            -1, -1, hands, -1, -1
        )
        pair_features = torch.cat(
            [hand_features, entity_features, relative, velocity, distance, valid],
            dim=-1,
        )
        interaction_tokens = self.pair_encoder(pair_features)
        contact_logits = self.contact_head(interaction_tokens).squeeze(-1)
        return {
            "interaction_tokens": interaction_tokens,
            "interaction_logits": contact_logits,
            "relative_hand_object_position": relative,
            "relative_hand_object_velocity": velocity,
        }
