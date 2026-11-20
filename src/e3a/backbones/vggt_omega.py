"""Thin adapter around the official VGGT-Ω implementation."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import signature
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import nn


class VGGTOmegaAdapter(nn.Module):
    """Expose official VGGT-Ω inference and its opt-in E3A feature API.

    The patched top-level VGGTOmega.forward owns all backbone/head orchestration.
    This adapter deliberately does not duplicate that forward implementation.
    """

    def __init__(
        self,
        checkpoint_path: str | PathLike[str] | None = None,
        *,
        model: nn.Module | None = None,
        strict: bool = True,
        freeze: bool = True,
        autocast: bool = False,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
    ) -> None:
        super().__init__()

        if model is None:
            try:
                from vggt_omega.models import VGGTOmega
            except ImportError as exc:
                raise ImportError(
                    "VGGT-Ω is not installed. Run "
                    "git submodule update --init --recursive and "
                    "pip install -e third_party/vggt-omega."
                ) from exc

            model = VGGTOmega(
                enable_camera=enable_camera,
                enable_depth=enable_depth,
                enable_alignment=enable_alignment,
                autocast=autocast,
            )

        self.model = model
        self.checkpoint_report: dict[str, Any] | None = None

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, strict=strict)

        if freeze:
            self.freeze()

    def load_checkpoint(
        self,
        checkpoint_path: str | PathLike[str],
        *,
        strict: bool = True,
    ) -> dict[str, list[str]]:
        """Load the raw upstream VGGT-Ω state dict without rewriting its keys."""
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"VGGT-Ω checkpoint does not exist: {path}")

        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) for key in state
        ):
            raise TypeError(
                "Expected the official VGGT-Ω checkpoint to be a string-keyed "
                "state-dict mapping."
            )

        incompatible = self.model.load_state_dict(state, strict=strict)
        report = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        self.checkpoint_report = report
        return report

    def freeze(self) -> None:
        """Freeze all upstream parameters."""
        self.model.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradients for all upstream parameters."""
        self.model.requires_grad_(True)

    @staticmethod
    def _ensure_batched(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            if images.shape[1] != 3:
                raise ValueError(
                    f"Expected [T, 3, H, W] input, got shape {tuple(images.shape)}"
                )
            return images.unsqueeze(0)
        if images.ndim == 5:
            if images.shape[2] != 3:
                raise ValueError(
                    f"Expected [B, T, 3, H, W] input, got shape {tuple(images.shape)}"
                )
            return images
        raise ValueError(
            "VGGT-Ω expects [T, 3, H, W] or [B, T, 3, H, W], "
            f"got {images.ndim} dimensions."
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the unmodified official inference contract."""
        images = self._ensure_batched(images)
        return self.model(images)

    def forward_with_features(
        self,
        images: torch.Tensor,
        *,
        extra_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Call the patched official forward with E3A tokens and fused features.

        Requires both pinned VGGT-Ω compatibility patches. The default upstream
        forward remains unchanged when these optional arguments are omitted.
        """
        images = self._ensure_batched(images)
        forward_parameters = signature(self.model.forward).parameters
        required_parameters = {"extra_tokens", "return_features"}
        if not required_parameters.issubset(forward_parameters):
            raise RuntimeError(
                "The pinned VGGT-Ω feature API patch is not applied. "
                "Run bash scripts/apply_patches.sh."
            )

        return self.model(
            images,
            extra_tokens=extra_tokens,
            return_features=True,
        )
