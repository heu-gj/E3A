import pytest
import torch

from e3a.data.hoi4d_supervision import HOI4DSupervisionBuilder, build_hoi4d_geometry_targets


def make_batch(*, intrinsics_valid=True):
    batch_size, frames, tracks, height, width = 1, 2, 1, 2, 2
    depth = torch.full((batch_size, frames, 1, height, width), 2.0)
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    pixel_valid = torch.ones(batch_size, frames, height, width, dtype=torch.bool)
    depth_valid = torch.ones(batch_size, frames, 1, height, width, dtype=torch.bool)
    masks = torch.ones(batch_size, tracks, frames, height, width)
    ids = torch.full((batch_size, tracks, frames), 17, dtype=torch.int64)
    mask_valid = torch.ones(batch_size, tracks, frames, dtype=torch.bool)
    return {
        "depth": depth,
        "depth_valid": depth_valid,
        "depth_units": ["unknown"],
        "intrinsics": intrinsics,
        "intrinsics_valid": torch.tensor([intrinsics_valid], dtype=torch.bool),
        "entity_targets": {
            "masks": masks,
            "ids": ids,
            "mask_valid": mask_valid,
            "pixel_valid": pixel_valid,
            "track_counts": torch.tensor([1], dtype=torch.int64),
        },
    }


def test_geometry_builder_unprojects_depth_and_pools_per_track():
    targets = build_hoi4d_geometry_targets(make_batch())

    assert targets["centers_camera"].shape == (1, 1, 2, 3)
    assert targets["covariance_camera"].shape == (1, 1, 2, 3, 3)
    assert targets["valid"].tolist() == [[[True, True]]]
    assert targets["point_mass"].tolist() == [[[4.0, 4.0]]]
    assert targets["coordinate_frame"] == "camera"
    assert targets["units"] == ["unknown"]
    assert targets["metric_valid"].tolist() == [False]

    torch.testing.assert_close(
        targets["centers_camera"][0, 0, 0], torch.tensor([1.0, 1.0, 2.0])
    )
    expected_covariance = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(
        targets["covariance_camera"][0, 0, 0], expected_covariance
    )


def test_geometry_builder_excludes_invalid_depth_and_unknown_pixels():
    batch = make_batch()
    batch["depth_valid"][0, 0, 0, 0, 0] = False
    batch["entity_targets"]["pixel_valid"][0, 1, 1, 1] = False

    targets = HOI4DSupervisionBuilder()(batch)["geometry_targets"]
    assert targets["valid"].all()
    assert targets["point_mass"].tolist() == [[[3.0, 3.0]]]
    torch.testing.assert_close(
        targets["centers_camera"][0, 0, 0],
        torch.tensor([4.0 / 3.0, 4.0 / 3.0, 2.0]),
    )


def test_geometry_builder_fails_closed_when_intrinsics_are_unavailable():
    targets = build_hoi4d_geometry_targets(make_batch(intrinsics_valid=False))
    assert not targets["valid"].any()
    assert not targets["metric_valid"].any()


def test_geometry_builder_does_not_create_targets_for_unannotated_tracks():
    batch = make_batch()
    batch["entity_targets"]["mask_valid"].zero_()
    targets = build_hoi4d_geometry_targets(batch)
    assert not targets["valid"].any()
    assert not targets["point_mass"].any()


def test_geometry_builder_rejects_singular_valid_intrinsics():
    batch = make_batch()
    batch["intrinsics"][0, 2] = 0
    with pytest.raises(ValueError, match="invertible"):
        build_hoi4d_geometry_targets(batch)
