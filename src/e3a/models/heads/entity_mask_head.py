"""Native query-to-mask head for E3A.

Each clip-level entity slot predicts one mask per observed frame. The same
slot index is supervised against the same track across the clip by the
clip-consistent VidEoMT matcher; no external segmentation model is required
in the online forward pass.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EntityMaskHead(nn.Module):
    """Predict masks by matching entity queries against dense VGGT-Ω features."""

    def __init__(
        self,
        *,
        entity_dim: int = 2048,
        pixel_dim: int = 256,
        mask_dim: int = 256,
        objectness_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(entity_dim, pixel_dim, mask_dim, objectness_hidden_dim) <= 0:
            raise ValueError("All feature dimensions must be positive.")

        self.entity_dim = int(entity_dim)
        self.pixel_dim = int(pixel_dim)
        self.mask_dim = int(mask_dim)

        self.entity_norm = nn.LayerNorm(entity_dim)
        self.entity_projection = MLP(entity_dim, mask_dim, mask_dim)
        self.pixel_projection = nn.Sequential(
            nn.Conv2d(pixel_dim, mask_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, mask_dim),
        )
        self.objectness_head = MLP(
            entity_dim,
            objectness_hidden_dim,
            1,
        )

    def forward(
        self,
        entity_tokens: torch.Tensor,
        fused_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if entity_tokens.ndim != 4:
            raise ValueError(
                "entity_tokens must have shape [B,T,K,D], "
                f"got {tuple(entity_tokens.shape)}"
            )
        if fused_features.ndim != 5:
            raise ValueError(
                "fused_features must have shape [B,T,C,H,W], "
                f"got {tuple(fused_features.shape)}"
            )
        if tuple(entity_tokens.shape[:2]) != tuple(fused_features.shape[:2]):
            raise ValueError("Entity tokens and dense features must align in [B,T].")
        if entity_tokens.shape[-1] != self.entity_dim:
            raise ValueError(
                f"Expected entity_dim={self.entity_dim}, got {entity_tokens.shape[-1]}"
            )
        if fused_features.shape[2] != self.pixel_dim:
            raise ValueError(
                f"Expected pixel_dim={self.pixel_dim}, got {fused_features.shape[2]}"
            )
        if entity_tokens.device != fused_features.device:
            raise ValueError("Entity tokens and dense features must share a device.")

        batch, frames, slots, _ = entity_tokens.shape
        _, _, channels, height, width = fused_features.shape
        flat_entities = entity_tokens.reshape(batch * frames, slots, self.entity_dim)
        flat_pixels = fused_features.reshape(
            batch * frames, channels, height, width
        )

        # Normalize mixed precision VGGT-Ω features to the head parameter dtype.
        head_dtype = self.entity_projection.layers[0].weight.dtype
        flat_entities = flat_entities.to(dtype=head_dtype)
        flat_pixels = flat_pixels.to(dtype=head_dtype)

        queries = self.entity_projection(self.entity_norm(flat_entities))
        pixels = self.pixel_projection(flat_pixels)
        mask_logits = torch.einsum("bkd,bdhw->bkhw", queries, pixels)
        mask_logits = mask_logits / math.sqrt(self.mask_dim)
        mask_logits = mask_logits.reshape(batch, frames, slots, height, width)

        objectness_logits = self.objectness_head(
            self.entity_norm(flat_entities)
        ).squeeze(-1)
        objectness_logits = objectness_logits.reshape(batch, frames, slots)

        return {
            "mask_logits": mask_logits,
            "objectness_logits": objectness_logits,
        }
