"""E3A representation encoder built on the official VGGT-Ω model."""

from __future__ import annotations

import torch
from torch import nn

from e3a.backbones.vggt_omega import VGGTOmegaAdapter
from e3a.models.token_bank import E3ATokenBank


class E3AEncoder(nn.Module):
    """Inject E3A tokens into VGGT-Ω and expose shared geometry/entity features.

    This is the representation stage, not the complete action-anticipation
    model. Mask, interaction, and future prediction heads are added only after
    dataset contracts and the feature path have passed validation.
    """

    def __init__(
        self,
        backbone: VGGTOmegaAdapter,
        token_bank: E3ATokenBank,
    ) -> None:
        super().__init__()
        if backbone.model.aggregator.patch_token_start <= 0:
            raise ValueError("Unexpected VGGT-Ω camera/register prefix length.")
        if token_bank.embed_dim != backbone.model.aggregator.camera_token.shape[-1]:
            raise ValueError(
                "TokenBank embed_dim must equal the VGGT-Ω Aggregator embedding dimension."
            )
        self.backbone = backbone
        self.token_bank = token_bank

    def forward(
        self,
        images: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError("images must have shape [B,T,3,H,W] or [T,3,H,W].")
        if timestamps.ndim == 1:
            timestamps = timestamps.unsqueeze(0)
        if timestamps.shape != images.shape[:2]:
            raise ValueError(
                "timestamps must match the image batch/frame dimensions: "
                f"images={tuple(images.shape[:2])}, timestamps={tuple(timestamps.shape)}"
            )

        extra_tokens = self.token_bank(timestamps)
        output = self.backbone.forward_with_features(
            images,
            extra_tokens=extra_tokens,
        )
        encoded_tokens = output.pop("extra_tokens")
        output.update(self.token_bank.split(encoded_tokens))
        return output
