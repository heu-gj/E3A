"""Entity-grounded future target, action, and time prediction."""

from __future__ import annotations

import math

import torch
from torch import nn


class AnticipationHead(nn.Module):
    """Predict the next target slot, action class, and time-to-event."""

    def __init__(
        self,
        *,
        embed_dim: int = 2048,
        interaction_dim: int = 256,
        num_actions: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if min(embed_dim, interaction_dim, num_actions, hidden_dim) <= 0:
            raise ValueError(
                "Anticipation head dimensions and num_actions must be positive."
            )
        self.embed_dim = int(embed_dim)
        self.num_actions = int(num_actions)
        self.future_projection = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim)
        )
        self.entity_projection = nn.Sequential(
            nn.LayerNorm(embed_dim + interaction_dim),
            nn.Linear(embed_dim + interaction_dim, embed_dim),
        )
        context_dim = embed_dim + interaction_dim
        self.no_target_head = nn.Linear(context_dim, 1)
        self.action_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.time_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        *,
        future_tokens: torch.Tensor,
        persistent_entity_tokens: torch.Tensor,
        interaction_tokens: torch.Tensor,
        interaction_logits: torch.Tensor,
        objectness_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if future_tokens.ndim != 4 or future_tokens.shape[2] != 1:
            raise ValueError("future_tokens must have shape [B,T,1,D].")
        if persistent_entity_tokens.ndim != 3:
            raise ValueError("persistent_entity_tokens must have shape [B,K,D].")
        if interaction_tokens.ndim != 5:
            raise ValueError("interaction_tokens must have shape [B,T,H,K,C].")
        if interaction_logits.shape != interaction_tokens.shape[:-1]:
            raise ValueError("interaction_logits must align with interaction_tokens.")
        expected_objectness_shape = (
            persistent_entity_tokens.shape[0],
            future_tokens.shape[1],
            persistent_entity_tokens.shape[1],
        )
        if objectness_logits.shape != expected_objectness_shape:
            raise ValueError("objectness_logits must have shape [B,T,K].")

        future_tokens = future_tokens.to(
            dtype=self.future_projection[1].weight.dtype
        )
        persistent_entity_tokens = persistent_entity_tokens.to(
            dtype=self.entity_projection[1].weight.dtype
        )
        interaction_tokens = interaction_tokens.to(
            dtype=self.entity_projection[1].weight.dtype
        )
        interaction_logits = interaction_logits.float()
        objectness_logits = objectness_logits.float()

        future = future_tokens.squeeze(2).mean(dim=1)
        interaction_by_entity = interaction_tokens.mean(dim=(1, 2))
        interaction_global = interaction_tokens.mean(dim=(1, 2, 3))
        context = torch.cat([future, interaction_global], dim=-1)

        entity_features = torch.cat(
            [persistent_entity_tokens, interaction_by_entity], dim=-1
        )
        entity_keys = self.entity_projection(entity_features)
        query = self.future_projection(future)
        target_scores = torch.einsum("bd,bkd->bk", query, entity_keys)
        target_scores = target_scores / math.sqrt(self.embed_dim)

        contact_prior = interaction_logits.sigmoid().mean(dim=(1, 2))
        objectness_prior = objectness_logits.sigmoid().mean(dim=1)
        target_scores = (
            target_scores
            + torch.logit(contact_prior.clamp(1e-4, 1 - 1e-4))
            + torch.logit(objectness_prior.clamp(1e-4, 1 - 1e-4))
        )
        target_entity_logits = torch.cat(
            [self.no_target_head(context), target_scores], dim=-1
        )

        time_parameters = self.time_head(context)
        return {
            "target_entity_logits": target_entity_logits,
            "action_logits": self.action_head(context),
            "time_log_mean": time_parameters[..., 0],
            "time_log_scale": time_parameters[..., 1].clamp(-5.0, 3.0),
        }
