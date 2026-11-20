"""Thin adapter around the official VGGT-Ω implementation.

This module intentionally does not reimplement VGGT-Ω. The upstream package is
installed from the pinned third_party/vggt-omega submodule.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from inspect import signature
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import nn


class VGGTOmegaAdapter(nn.Module):
    """Expose official VGGT-Ω inference and an opt-in E3A feature path."""

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
        """Load an upstream state dict without silently rewriting its keys."""
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
        """Run the unmodified official top-level forward and pass outputs through."""
        self._ensure_batched(images)
        return self.model(images)

    def forward_with_features(
        self,
        images: torch.Tensor,
        *,
        extra_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run official submodules and additionally expose E3A features.

        Requires the pinned patch applied by scripts/apply_patches.sh.
        CameraHead sees only the original camera/register prefix; DenseHead sees
        the expanded prefix so E3A tokens are not mistaken for image patches.
        """
        images = self._ensure_batched(images)

        if not hasattr(self.model, "aggregator") or not hasattr(
            self.model, "dense_head"
        ):
            raise RuntimeError("The injected model is not a compatible VGGTOmega model.")
        if self.model.dense_head is None:
            raise RuntimeError("DenseHead is disabled; fused features are unavailable.")

        aggregator = self.model.aggregator
        if "extra_tokens" not in signature(aggregator.forward).parameters:
            raise RuntimeError(
                "The VGGT-Ω Aggregator patch is not applied. Run bash scripts/apply_patches.sh."
            )
        if "return_features" not in signature(self.model.dense_head.forward).parameters:
            raise RuntimeError(
                "The VGGT-Ω DenseHead patch is not applied. Run bash scripts/apply_patches.sh."
            )

        original_prefix = aggregator.patch_token_start

        if self.model.autocast and images.device.type == "cuda":
            amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            aggregator_context = torch.autocast(
                device_type="cuda", dtype=amp_dtype
            )
        else:
            aggregator_context = nullcontext()

        with aggregator_context:
            aggregated_tokens_list, expanded_prefix = aggregator(
                images, extra_tokens=extra_tokens
            )

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer.")

        predictions: dict[str, torch.Tensor] = {
            "camera_and_register_tokens": final_tokens[
                :, :, :original_prefix
            ].contiguous()
        }
        if expanded_prefix > original_prefix:
            predictions["extra_tokens"] = final_tokens[
                :, :, original_prefix:expanded_prefix
            ].contiguous()

        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.model.camera_head is not None:
                predictions["pose_enc"] = self.model.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=original_prefix,
                )

            dense_output = self.model.dense_head(
                aggregated_tokens_list,
                images=images,
                patch_token_start=expanded_prefix,
                return_features=True,
            )
            if len(dense_output) != 3:
                raise RuntimeError(
                    "Patched DenseHead must return depth, depth_conf, and fused_features."
                )
            depth, depth_conf, fused_features = dense_output
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
            predictions["fused_features"] = fused_features

            if self.model.text_alignment_head is not None:
                predictions.update(
                    self.model.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=expanded_prefix,
                    )
                )

        if not self.model.training:
            predictions["images"] = images
        return predictions
