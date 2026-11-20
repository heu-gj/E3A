"""Differentiable mask-weighted pooling of VGGT-Ω world-space points."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pool_entity_geometry(
    mask_logits: torch.Tensor,
    points_world: torch.Tensor,
    depth: torch.Tensor,
    *,
    min_mass: float = 1.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Pool per-entity 3D centers and covariance from soft masks.

    Args:
        mask_logits: Entity mask logits [B,T,K,Hm,Wm], typically at 1/4 scale.
        points_world: Official VGGT-Ω unprojected points [B,T,H,W,3].
        depth: VGGT-Ω depth [B,T,H,W] or [B,T,H,W,1].
        min_mass: Minimum sum of valid soft-mask weights for a valid entity.
        eps: Denominator floor used for numerically stable pooling.

    Returns:
        entity_centers_world: [B,T,K,3].
        entity_covariance_world: [B,T,K,3,3].
        entity_geometry_valid: [B,T,K] boolean.
        entity_mask_mass: [B,T,K].

    Depth confidence is intentionally not used here: VGGT-Ω's depth_conf is a
    learned positive loss weight, not a calibrated probability in [0,1].
    """
    if mask_logits.ndim != 5:
        raise ValueError(
            f"mask_logits must have shape [B,T,K,H,W], got {tuple(mask_logits.shape)}"
        )
    if points_world.ndim != 5 or points_world.shape[-1] != 3:
        raise ValueError(
            "points_world must have shape [B,T,H,W,3], "
            f"got {tuple(points_world.shape)}"
        )
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth_values = depth[..., 0]
    elif depth.ndim == 4:
        depth_values = depth
    else:
        raise ValueError(
            "depth must have shape [B,T,H,W] or [B,T,H,W,1], "
            f"got {tuple(depth.shape)}"
        )

    batch_size, num_frames, num_entities, mask_height, mask_width = mask_logits.shape
    expected_prefix = (batch_size, num_frames)
    if tuple(points_world.shape[:2]) != expected_prefix:
        raise ValueError("mask_logits and points_world batch/frame dimensions must match.")
    if tuple(depth_values.shape[:2]) != expected_prefix:
        raise ValueError("mask_logits and depth batch/frame dimensions must match.")
    if tuple(points_world.shape[2:4]) != tuple(depth_values.shape[2:4]):
        raise ValueError("points_world and depth spatial dimensions must match.")
    if not math.isfinite(min_mass) or not math.isfinite(eps) or min_mass < 0 or eps <= 0:
        raise ValueError("min_mass must be finite/non-negative and eps finite/positive.")
    if not torch.isfinite(mask_logits).all():
        raise ValueError("mask_logits must contain only finite values.")
    if (
        mask_logits.device != points_world.device
        or mask_logits.device != depth.device
    ):
        raise ValueError("mask_logits, points_world, and depth must be on the same device.")

    height, width = depth_values.shape[-2:]
    probabilities = mask_logits.float().sigmoid()
    if (mask_height, mask_width) != (height, width):
        probabilities = F.interpolate(
            probabilities.reshape(
                batch_size * num_frames * num_entities,
                1,
                mask_height,
                mask_width,
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_frames, num_entities, height, width)

    points = points_world.float()
    depth_values = depth_values.float()
    valid = (
        torch.isfinite(depth_values)
        & (depth_values > 0)
        & torch.isfinite(points).all(dim=-1)
    )
    weights = probabilities * valid.unsqueeze(2).to(probabilities.dtype)
    mass = weights.sum(dim=(-2, -1))
    geometry_valid = mass >= min_mass

    safe_points = torch.where(valid.unsqueeze(-1), points, torch.zeros_like(points))
    point_grid = safe_points.unsqueeze(2)
    weighted_sum = (weights.unsqueeze(-1) * point_grid).sum(dim=(-3, -2))
    centers = weighted_sum / mass.clamp_min(eps).unsqueeze(-1)

    offsets = point_grid - centers[..., None, None, :]
    covariance = torch.einsum(
        "btkhw,btkhwi,btkhwj->btkij",
        weights,
        offsets,
        offsets,
    ) / mass.clamp_min(eps)[..., None, None]

    centers = torch.where(
        geometry_valid.unsqueeze(-1),
        centers,
        torch.zeros_like(centers),
    )
    covariance = torch.where(
        geometry_valid[..., None, None],
        covariance,
        torch.zeros_like(covariance),
    )

    return {
        "entity_centers_world": centers,
        "entity_covariance_world": covariance,
        "entity_geometry_valid": geometry_valid,
        "entity_mask_mass": mass,
    }
