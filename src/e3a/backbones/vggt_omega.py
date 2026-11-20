"""Thin adapter around the official VGGT-Ω implementation.

This module intentionally does not reimplement VGGT-Ω. The upstream package is
installed from the pinned third_party/vggt-omega submodule.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import nn


class VGGTOmegaAdapter(nn.Module):
    """Expose the official VGGTOmega model through a stable E3A boundary.

    The official model remains responsible for preprocessing, aggregation,
    camera prediction, dense depth prediction, and output naming.
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
        """Load an upstream checkpoint without silently rewriting its keys.

        The released VGGT-Ω quick-start loads a state dict directly with
        load_state_dict. This adapter deliberately does not guess whether a
        checkpoint is wrapped in keys such as model or state_dict.
        """
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

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the official VGGT-Ω forward pass.

        Accepted input layouts match upstream: [T,3,H,W] or [B,T,3,H,W].
        Returned keys and tensor semantics are passed through unchanged.
        """
        if images.ndim == 4:
            if images.shape[1] != 3:
                raise ValueError(
                    f"Expected [T, 3, H, W] input, got shape {tuple(images.shape)}"
                )
        elif images.ndim == 5:
            if images.shape[2] != 3:
                raise ValueError(
                    f"Expected [B, T, 3, H, W] input, got shape {tuple(images.shape)}"
                )
        else:
            raise ValueError(
                "VGGT-Ω expects [T, 3, H, W] or [B, T, 3, H, W], "
                f"got {images.ndim} dimensions."
            )

        return self.model(images)
