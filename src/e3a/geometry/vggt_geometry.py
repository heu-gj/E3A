"""Geometry helpers that delegate to official VGGT-Ω utilities."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch

from vggt_omega.utils.pose_enc import encoding_to_camera


@lru_cache(maxsize=1)
def _official_unprojection_module() -> ModuleType:
    """Load the exact unprojection helper shipped in VGGT-Ω's training tree.

    VGGT-Ω's pyproject intentionally packages only vggt_omega, while its
    training/losses/unprojection.py contains the official depth unprojection
    implementation. Load that source file directly rather than copying or
    reimplementing the geometry routine in E3A.
    """
    repo_root = Path(__file__).resolve().parents[3]
    source = (
        repo_root
        / "third_party"
        / "vggt-omega"
        / "training"
        / "losses"
        / "unprojection.py"
    )
    if not source.is_file():
        raise FileNotFoundError(
            "The pinned VGGT-Ω training source is missing. Run "
            "git submodule update --init third_party/vggt-omega."
        )

    spec = spec_from_file_location("e3a_vggt_omega_unprojection", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official VGGT-Ω unprojection helper: {source}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_and_unproject(
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Decode official camera predictions and unproject depth into world points.

    Args:
        depth: VGGT-Ω depth, shape [B,T,H,W] or [B,T,H,W,1].
        pose_enc: VGGT-Ω pose encoding, shape [B,T,9].
        image_size: (height, width) used by the camera head. Defaults to the
            spatial dimensions of depth.

    Returns:
        A dictionary containing world-space points [B,T,H,W,3], the decoded
        camera-from-world extrinsics [B,T,3,4], and intrinsics [B,T,3,3].

    The returned extrinsics follow VGGT-Ω's OpenCV camera-from-world convention.
    The official training unprojection helper performs the inverse transform
    and returns world-space points.
    """
    if depth.ndim not in (4, 5):
        raise ValueError(
            "depth must have shape [B,T,H,W] or [B,T,H,W,1], "
            f"got {tuple(depth.shape)}"
        )
    if pose_enc.ndim != 3 or pose_enc.shape[-1] != 9:
        raise ValueError(
            f"pose_enc must have shape [B,T,9], got {tuple(pose_enc.shape)}"
        )
    if tuple(depth.shape[:2]) != tuple(pose_enc.shape[:2]):
        raise ValueError(
            "depth and pose_enc batch/frame dimensions must match: "
            f"{tuple(depth.shape[:2])} vs {tuple(pose_enc.shape[:2])}"
        )

    height, width = depth.shape[2:4]
    if image_size is None:
        image_size = (height, width)
    if tuple(image_size) != (height, width):
        raise ValueError(
            "image_size must match the depth resolution because VGGT-Ω camera "
            f"intrinsics are decoded for that size; got {image_size}, depth={(height, width)}"
        )

    extrinsics, intrinsics = encoding_to_camera(
        pose_enc,
        image_size,
        build_intrinsics=True,
    )
    unproject = _official_unprojection_module().unproject_depth_to_points_torch_batch
    points_world = unproject(depth, extrinsics, intrinsics)

    return {
        "points_world": points_world,
        "extrinsics_camera_from_world": extrinsics,
        "intrinsics": intrinsics,
    }
