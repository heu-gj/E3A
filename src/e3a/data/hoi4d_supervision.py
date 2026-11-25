"""Canonical HOI4D supervision construction from released RGB-D annotations.

This module derives camera-frame object geometry from source depth, camera
intrinsics, entity masks, and explicit annotation-validity masks. Depth units
remain untouched: coordinates use the source depth unit and the training loss
is scale-invariant within each clip. No metric-unit claim is made here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


class HOI4DSupervisionBuilder:
    """Build validity-aware geometry targets for a collated HOI4D batch.

    Expected batch tensors:
      depth: [B,T,1,H,W], raw source depth values
      depth_valid: [B,T,1,H,W], optional; otherwise derived from depth
      intrinsics: [B,3,3]
      intrinsics_valid: [B], required for geometry supervision
      entity_targets.masks: [B,N,T,H,W]
      entity_targets.ids: [B,N,T]
      entity_targets.mask_valid: [B,N,T]
      entity_targets.pixel_valid: [B,T,H,W]
      entity_targets.track_counts: [B]

    Returned geometry targets are track-major:
      centers_camera: [B,N,T,3]
      covariance_camera: [B,N,T,3,3]
      valid: [B,N,T]
      point_mass: [B,N,T]

    Object-pose records are intentionally not consumed: their physical units
    and coordinate-frame convention are not yet part of the canonical target
    contract.
    """

    def __init__(self, *, min_points: float = 1.0, eps: float = 1e-6) -> None:
        if not math.isfinite(min_points) or min_points < 0:
            raise ValueError("min_points must be finite and non-negative.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.min_points = float(min_points)
        self.eps = float(eps)

    @staticmethod
    def _require_tensor(
        mapping: Mapping[str, Any],
        key: str,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        value = mapping.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"HOI4D geometry builder requires tensor '{key}'.")
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f"HOI4D geometry target '{key}' must use {dtype}.")
        return value

    def build_geometry_targets(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Derive per-track/per-frame 3D centers and covariance in camera coordinates."""
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping.")
        entity_targets = batch.get("entity_targets")
        if not isinstance(entity_targets, Mapping):
            raise ValueError("batch must contain an entity_targets mapping.")

        depth = self._require_tensor(batch, "depth").to(dtype=torch.float32)
        intrinsics = self._require_tensor(batch, "intrinsics").to(dtype=torch.float32)
        intrinsics_valid = self._require_tensor(
            batch, "intrinsics_valid", dtype=torch.bool
        )
        masks = self._require_tensor(entity_targets, "masks").to(dtype=torch.float32)
        ids = self._require_tensor(entity_targets, "ids")
        mask_valid = self._require_tensor(
            entity_targets, "mask_valid", dtype=torch.bool
        )
        pixel_valid = self._require_tensor(
            entity_targets, "pixel_valid", dtype=torch.bool
        )
        track_counts = self._require_tensor(entity_targets, "track_counts")

        if depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B,T,1,H,W].")
        batch_size, frames, _, height, width = depth.shape
        if intrinsics.shape != (batch_size, 3, 3):
            raise ValueError("intrinsics must have shape [B,3,3].")
        if intrinsics_valid.shape != (batch_size,):
            raise ValueError("intrinsics_valid must have shape [B].")
        if masks.ndim != 5 or tuple(masks.shape[0:1] + masks.shape[2:]) != (
            batch_size,
            frames,
            height,
            width,
        ):
            raise ValueError("entity masks must have shape [B,N,T,H,W] aligned with depth.")
        max_tracks = masks.shape[1]
        if ids.shape != (batch_size, max_tracks, frames):
            raise ValueError("entity IDs must have shape [B,N,T].")
        if mask_valid.shape != ids.shape:
            raise ValueError("mask_valid must have shape [B,N,T].")
        if pixel_valid.shape != (batch_size, frames, height, width):
            raise ValueError("pixel_valid must have shape [B,T,H,W].")
        if track_counts.shape != (batch_size,):
            raise ValueError("track_counts must have shape [B].")
        if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("entity IDs must use an integer dtype.")
        if track_counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("track_counts must use an integer dtype.")
        if not torch.isfinite(masks).all() or (
            masks.numel() and ((masks < 0).any() or (masks > 1).any())
        ):
            raise ValueError("entity masks must be finite values in [0,1].")

        depth_valid = batch.get("depth_valid")
        if depth_valid is None:
            depth_valid = torch.isfinite(depth) & (depth > 0)
        elif (
            not torch.is_tensor(depth_valid)
            or depth_valid.dtype != torch.bool
            or depth_valid.shape != depth.shape
        ):
            raise ValueError("depth_valid must be bool [B,T,1,H,W].")

        device = depth.device
        intrinsics = intrinsics.to(device=device)
        intrinsics_valid = intrinsics_valid.to(device=device)
        masks = masks.to(device=device)
        ids = ids.to(device=device, dtype=torch.int64)
        mask_valid = mask_valid.to(device=device)
        pixel_valid = pixel_valid.to(device=device)
        depth_valid = depth_valid.to(device=device)
        track_counts = track_counts.to(device=device, dtype=torch.int64)

        if (track_counts < 0).any() or (track_counts > max_tracks).any():
            raise ValueError("track_counts must lie between zero and the padded track count.")

        centers = torch.zeros(
            (batch_size, max_tracks, frames, 3), dtype=torch.float32, device=device
        )
        covariance = torch.zeros(
            (batch_size, max_tracks, frames, 3, 3),
            dtype=torch.float32,
            device=device,
        )
        geometry_valid = torch.zeros(
            (batch_size, max_tracks, frames), dtype=torch.bool, device=device
        )
        point_mass = torch.zeros(
            (batch_size, max_tracks, frames), dtype=torch.float32, device=device
        )

        rows, columns = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        pixel_grid = torch.stack(
            (columns, rows, torch.ones_like(columns)), dim=-1
        )

        for batch_index in range(batch_size):
            if not bool(intrinsics_valid[batch_index].item()):
                continue
            camera_matrix = intrinsics[batch_index]
            if not torch.isfinite(camera_matrix).all():
                raise ValueError("Valid HOI4D camera intrinsics must be finite.")
            determinant = torch.linalg.det(camera_matrix)
            if not torch.isfinite(determinant) or determinant.abs() <= self.eps:
                raise ValueError("Valid HOI4D camera intrinsics must be invertible.")
            inverse_intrinsics = torch.linalg.inv(camera_matrix)
            rays = torch.einsum("ij,hwj->hwi", inverse_intrinsics, pixel_grid)
            ray_z = rays[..., 2]
            if not torch.isfinite(rays).all() or (ray_z.abs() <= self.eps).any():
                raise ValueError("HOI4D intrinsics produced invalid camera rays.")
            rays = rays / ray_z.unsqueeze(-1)

            count = int(track_counts[batch_index].item())
            for frame_index in range(frames):
                source_depth = depth[batch_index, frame_index, 0]
                valid_pixels = (
                    depth_valid[batch_index, frame_index, 0]
                    & pixel_valid[batch_index, frame_index]
                    & torch.isfinite(source_depth)
                    & (source_depth > 0)
                )
                if not valid_pixels.any():
                    continue
                camera_points = rays * source_depth.unsqueeze(-1)
                camera_points = torch.where(
                    valid_pixels.unsqueeze(-1),
                    camera_points,
                    torch.zeros_like(camera_points),
                )

                for track_index in range(count):
                    if not bool(mask_valid[batch_index, track_index, frame_index].item()):
                        continue
                    if int(ids[batch_index, track_index, frame_index].item()) < 0:
                        continue
                    weights = (
                        masks[batch_index, track_index, frame_index]
                        * valid_pixels.to(dtype=torch.float32)
                    )
                    mass = weights.sum()
                    if not torch.isfinite(mass) or float(mass.item()) < self.min_points:
                        continue
                    center = (
                        weights.unsqueeze(-1) * camera_points
                    ).sum(dim=(0, 1)) / mass.clamp_min(self.eps)
                    offsets = camera_points - center
                    cov = torch.einsum(
                        "hw,hwi,hwj->ij", weights, offsets, offsets
                    ) / mass.clamp_min(self.eps)
                    if not torch.isfinite(center).all() or not torch.isfinite(cov).all():
                        raise FloatingPointError(
                            "Valid HOI4D geometry target contains non-finite values."
                        )
                    centers[batch_index, track_index, frame_index] = center
                    covariance[batch_index, track_index, frame_index] = cov
                    geometry_valid[batch_index, track_index, frame_index] = True
                    point_mass[batch_index, track_index, frame_index] = mass

        units = batch.get("depth_units")
        if not isinstance(units, (list, tuple)) or len(units) != batch_size:
            units = ["unknown"] * batch_size

        return {
            "centers_camera": centers,
            "covariance_camera": covariance,
            "valid": geometry_valid,
            "point_mass": point_mass,
            "metric_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "units": [str(unit) for unit in units],
            "coordinate_frame": "camera",
            "normalization": "scale_invariant_per_clip",
        }

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Return a shallow-copied batch enriched with canonical geometry targets."""
        result = dict(batch)
        result["geometry_targets"] = self.build_geometry_targets(batch)
        return result


def build_hoi4d_geometry_targets(
    batch: Mapping[str, Any],
    *,
    min_points: float = 1.0,
) -> dict[str, Any]:
    """Convenience wrapper for the HOI4D camera-frame geometry target builder."""
    return HOI4DSupervisionBuilder(min_points=min_points).build_geometry_targets(batch)
