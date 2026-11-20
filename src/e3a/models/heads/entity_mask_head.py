"""Entity mask head reusing SAM 3's official MaskPredictor implementation."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


def _register_sam3_package(name: str, package_path: Path) -> None:
    """Expose a narrow SAM 3 package path without running its model-builder __init__."""
    package_path = package_path.resolve()
    package = sys.modules.get(name)
    if package is None:
        package = ModuleType(name)
        package.__path__ = [str(package_path)]
        package.__package__ = name
        sys.modules[name] = package
        return

    existing_paths = {
        Path(path).resolve()
        for path in getattr(package, "__path__", [])
    }
    if package_path not in existing_paths:
        raise ImportError(
            f"{name} is already imported from a different location; "
            "E3A requires the pinned SAM 3 source."
        )


def _load_sam3_mask_predictor():
    repo_root = Path(__file__).resolve().parents[4]
    sam3_package_root = repo_root / "third_party" / "sam3" / "sam3"
    if not sam3_package_root.is_dir():
        try:
            spec = find_spec("sam3")
        except (ImportError, ValueError):
            spec = None
        locations = (
            list(spec.submodule_search_locations)
            if spec is not None and spec.submodule_search_locations is not None
            else []
        )
        if not locations:
            raise ImportError(
                "SAM 3 source is missing. Initialize third_party/sam3 or install "
                "the official SAM 3 package."
            )
        sam3_package_root = Path(locations[0]).resolve()

    # The root sam3/__init__.py imports the full model builder. E3A only needs
    # maskformer_segmentation.MaskPredictor and its local model_misc dependency.
    _register_sam3_package("sam3", sam3_package_root)
    _register_sam3_package("sam3.model", sam3_package_root / "model")

    from sam3.model.maskformer_segmentation import MaskPredictor

    return MaskPredictor


class EntityMaskHead(nn.Module):
    """Project VGGT-Ω entity queries and reuse SAM 3's query-to-mask predictor.

    SAM 3's MaskPredictor implementation is used directly. E3A only provides
    the feature-dimension adapter needed to connect VGGT-Ω's 2048-D entity
    tokens and 256-D DenseHead fused features to that upstream module.
    """

    def __init__(
        self,
        *,
        entity_dim: int = 2048,
        pixel_dim: int = 256,
        mask_dim: int = 256,
        mask_predictor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if entity_dim <= 0 or pixel_dim <= 0 or mask_dim <= 0:
            raise ValueError("entity_dim, pixel_dim, and mask_dim must be positive.")

        self.entity_dim = entity_dim
        self.pixel_dim = pixel_dim
        self.mask_dim = mask_dim
        self.entity_projection = nn.Linear(entity_dim, mask_dim)
        self.pixel_projection = (
            nn.Identity()
            if pixel_dim == mask_dim
            else nn.Conv2d(pixel_dim, mask_dim, kernel_size=1)
        )
        self.objectness_head = nn.Linear(entity_dim, 1)

        if mask_predictor is None:
            try:
                mask_predictor_cls = _load_sam3_mask_predictor()
            except ImportError as exc:
                raise ImportError(
                    "SAM 3 is required for EntityMaskHead. Initialize "
                    "third_party/sam3 or install the official SAM 3 package."
                ) from exc
            mask_predictor = mask_predictor_cls(
                hidden_dim=mask_dim,
                mask_dim=mask_dim,
            )

        self.mask_predictor = mask_predictor

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
            raise ValueError("Entity token and fused feature batch/frame dimensions must match.")
        if entity_tokens.shape[-1] != self.entity_dim:
            raise ValueError(
                f"Expected entity feature dimension {self.entity_dim}, "
                f"got {entity_tokens.shape[-1]}"
            )
        if fused_features.shape[2] != self.pixel_dim:
            raise ValueError(
                f"Expected fused feature dimension {self.pixel_dim}, "
                f"got {fused_features.shape[2]}"
            )
        if entity_tokens.device != fused_features.device:
            raise ValueError("Entity tokens and fused features must be on the same device.")

        batch_size, num_frames, num_entities, _ = entity_tokens.shape
        _, _, pixel_channels, height, width = fused_features.shape

        # VGGT-Ω may emit bfloat16 cached tokens under its internal autocast,
        # while the E3A/SAM 3 heads are normally float32. Normalize inputs to
        # the projection parameter dtype; outer AMP can still autocast ops.
        feature_dtype = self.entity_projection.weight.dtype
        entity_features = entity_tokens.to(dtype=feature_dtype)
        queries = self.entity_projection(entity_features)
        queries = queries.reshape(batch_size * num_frames, num_entities, -1)

        pixels = fused_features.to(dtype=feature_dtype).reshape(
            batch_size * num_frames,
            pixel_channels,
            height,
            width,
        )
        pixels = self.pixel_projection(pixels)

        mask_logits = self.mask_predictor(queries, pixels)
        mask_logits = mask_logits.reshape(
            batch_size,
            num_frames,
            num_entities,
            mask_logits.shape[-2],
            mask_logits.shape[-1],
        )
        objectness_logits = self.objectness_head(entity_features).squeeze(-1)

        return {
            "mask_logits": mask_logits,
            "objectness_logits": objectness_logits,
        }
