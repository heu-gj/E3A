"""Learnable E3A entity, hand, and future tokens."""

from __future__ import annotations

import torch
from torch import nn


class E3ATokenBank(nn.Module):
    """Create per-frame entity/hand/future tokens for the VGGT-Ω Aggregator.

    Timestamps are seconds relative to the start of each clip and have shape
    [B,T]. Their embedding is added to every token so inter-frame attention has
    an explicit temporal signal. Entity count is intentionally required rather
    than silently assuming a dataset-specific number.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_entities: int,
        num_hands: int = 2,
        time_scale_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_entities <= 0 or num_hands <= 0:
            raise ValueError("embed_dim, num_entities, and num_hands must be positive.")
        if time_scale_seconds <= 0:
            raise ValueError("time_scale_seconds must be positive.")

        self.embed_dim = embed_dim
        self.num_entities = num_entities
        self.num_hands = num_hands
        self.time_scale_seconds = float(time_scale_seconds)

        self.entity_tokens = nn.Parameter(torch.empty(1, 1, num_entities, embed_dim))
        self.hand_tokens = nn.Parameter(torch.empty(1, 1, num_hands, embed_dim))
        self.future_token = nn.Parameter(torch.empty(1, 1, 1, embed_dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        nn.init.normal_(self.entity_tokens, std=0.02)
        nn.init.normal_(self.hand_tokens, std=0.02)
        nn.init.normal_(self.future_token, std=0.02)

        self.entity_slice = slice(0, num_entities)
        self.hand_slice = slice(num_entities, num_entities + num_hands)
        self.future_slice = slice(num_entities + num_hands, num_entities + num_hands + 1)

    @property
    def num_tokens(self) -> int:
        return self.num_entities + self.num_hands + 1

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        if timestamps.ndim != 2:
            raise ValueError(
                f"timestamps must have shape [B,T], got {tuple(timestamps.shape)}"
            )
        if timestamps.shape[1] == 0:
            raise ValueError("timestamps must contain at least one frame.")
        if not torch.isfinite(timestamps).all():
            raise ValueError("timestamps must contain only finite values.")

        timestamps = timestamps.to(
            device=self.entity_tokens.device,
            dtype=self.entity_tokens.dtype,
        )
        relative_seconds = timestamps - timestamps[:, :1]
        time_features = self.time_embedding(
            (relative_seconds / self.time_scale_seconds).unsqueeze(-1)
        )

        batch_size, num_frames = timestamps.shape
        base_tokens = torch.cat(
            [self.entity_tokens, self.hand_tokens, self.future_token], dim=2
        )
        base_tokens = base_tokens.expand(batch_size, num_frames, -1, -1)
        return base_tokens + time_features.unsqueeze(2)

    def split(self, encoded_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if encoded_tokens.ndim != 4:
            raise ValueError(
                "encoded_tokens must have shape [B,T,N,D], "
                f"got {tuple(encoded_tokens.shape)}"
            )
        if encoded_tokens.shape[2] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens, got {encoded_tokens.shape[2]}"
            )
        return {
            "entity_tokens": encoded_tokens[:, :, self.entity_slice],
            "hand_tokens": encoded_tokens[:, :, self.hand_slice],
            "future_tokens": encoded_tokens[:, :, self.future_slice],
        }
