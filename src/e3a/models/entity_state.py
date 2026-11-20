"""Clip-level persistent entity state pooling."""

from __future__ import annotations

import torch
from torch import nn


class PersistentEntityPool(nn.Module):
    """Aggregate each temporally consistent slot into one clip-level state.

    Slot identity is clip-local: slot k is matched to one GT track across the
    observed clip by the training matcher. This module pools its per-frame
    features for anticipation without introducing a second tracking network.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        self.score = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    def forward(
        self,
        entity_tokens: torch.Tensor,
        *,
        entity_centers_world: torch.Tensor | None = None,
        entity_geometry_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if entity_tokens.ndim != 4:
            raise ValueError(
                "entity_tokens must have shape [B,T,K,D], "
                f"got {tuple(entity_tokens.shape)}"
            )
        features = entity_tokens.to(dtype=self.score[1].weight.dtype)
        scores = self.score(features).squeeze(-1)
        temporal_weights = scores.softmax(dim=1)
        persistent_tokens = (temporal_weights.unsqueeze(-1) * features).sum(dim=1)
        result = {
            "persistent_entity_tokens": persistent_tokens,
            "entity_temporal_weights": temporal_weights,
        }

        if entity_centers_world is not None:
            if entity_centers_world.shape != (*entity_tokens.shape[:3], 3):
                raise ValueError("entity_centers_world must have shape [B,T,K,3].")
            if entity_geometry_valid is None:
                valid = torch.ones(
                    entity_tokens.shape[:3],
                    dtype=torch.bool,
                    device=entity_tokens.device,
                )
            else:
                if entity_geometry_valid.shape != entity_tokens.shape[:3]:
                    raise ValueError("entity_geometry_valid must have shape [B,T,K].")
                valid = entity_geometry_valid.bool()

            weights = temporal_weights * valid.to(temporal_weights.dtype)
            denominator = weights.sum(dim=1)
            normalized = weights / denominator.clamp_min(1e-6).unsqueeze(1)
            centers = (normalized.unsqueeze(-1) * entity_centers_world).sum(dim=1)
            geometry_valid = denominator > 0
            centers = torch.where(
                geometry_valid.unsqueeze(-1), centers, torch.zeros_like(centers)
            )
            result.update(
                {
                    "persistent_entity_centers_world": centers,
                    "persistent_entity_geometry_valid": geometry_valid,
                }
            )
        return result
