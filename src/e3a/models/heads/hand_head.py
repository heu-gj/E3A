"""Lightweight 3D hand-pose head driven by VGGT-Ω hand registers."""

from __future__ import annotations

import torch
from torch import nn


class HandPoseHead(nn.Module):
    """Predict two camera-frame hands with 21 joints each.

    Joint order is wrist, thumb(1..4), index(1..4), middle(1..4),
    ring(1..4), pinky(1..4). Dataset adapters must convert source annotations
    into this order and the camera coordinate frame before supervision.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 2048,
        hidden_dim: int = 512,
        num_hands: int = 2,
        num_joints: int = 21,
    ) -> None:
        super().__init__()
        if min(embed_dim, hidden_dim, num_hands, num_joints) <= 0:
            raise ValueError("Hand head dimensions must be positive.")
        self.num_hands = int(num_hands)
        self.num_joints = int(num_joints)
        self.pose_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_joints),
        )

    def forward(self, hand_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if hand_tokens.ndim != 4:
            raise ValueError(
                "hand_tokens must have shape [B,T,H,D], "
                f"got {tuple(hand_tokens.shape)}"
            )
        if hand_tokens.shape[2] != self.num_hands:
            raise ValueError(
                f"Expected {self.num_hands} hand tokens, got {hand_tokens.shape[2]}"
            )
        dtype = self.pose_head[1].weight.dtype
        features = hand_tokens.to(dtype=dtype)
        batch, frames, hands, _ = features.shape
        keypoints = self.pose_head(features).reshape(
            batch, frames, hands, self.num_joints, 3
        )
        visibility = self.visibility_head(features)
        return {
            "hand_keypoints_camera": keypoints,
            "hand_joint_visibility_logits": visibility,
        }
