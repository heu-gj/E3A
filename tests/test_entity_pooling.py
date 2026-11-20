import torch

from e3a.geometry.entity_pooling import pool_entity_geometry


def test_pooling_computes_world_center_and_covariance():
    points = torch.tensor(
        [[[
            [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
            [[0.0, 2.0, 1.0], [2.0, 2.0, 1.0]],
        ]]]
    )
    depth = torch.ones(1, 1, 2, 2)
    logits = torch.full((1, 1, 1, 2, 2), 10.0)

    output = pool_entity_geometry(logits, points, depth, min_mass=1.0)

    assert output["entity_centers_world"].shape == (1, 1, 1, 3)
    assert output["entity_covariance_world"].shape == (1, 1, 1, 3, 3)
    assert output["entity_geometry_valid"].item()
    torch.testing.assert_close(
        output["entity_centers_world"],
        torch.tensor([[[[1.0, 1.0, 1.0]]]]),
        atol=1e-4,
        rtol=1e-4,
    )


def test_pooling_marks_empty_or_invalid_depth_entities_invalid():
    points = torch.ones(1, 1, 2, 2, 3)
    depth = torch.zeros(1, 1, 2, 2)
    logits = torch.full((1, 1, 2, 1, 1), -100.0)

    output = pool_entity_geometry(logits, points, depth, min_mass=0.5)

    assert not output["entity_geometry_valid"].any()
    assert torch.count_nonzero(output["entity_centers_world"]) == 0
    assert torch.count_nonzero(output["entity_covariance_world"]) == 0
    assert torch.count_nonzero(output["entity_mask_mass"]) == 0


def test_pooling_upsamples_mask_logits_to_depth_resolution():
    points = torch.zeros(1, 1, 4, 4, 3)
    points[..., 2] = 1.0
    depth = torch.ones(1, 1, 4, 4)
    logits = torch.full((1, 1, 1, 1, 1), 10.0)

    output = pool_entity_geometry(logits, points, depth, min_mass=1.0)

    assert output["entity_geometry_valid"].item()
    assert output["entity_mask_mass"].item() > 15.0
    torch.testing.assert_close(
        output["entity_centers_world"],
        torch.tensor([[[[0.0, 0.0, 1.0]]]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_pooling_ignores_non_finite_world_points():
    points = torch.ones(1, 1, 2, 2, 3)
    points[0, 0, 0, 0] = float("nan")
    depth = torch.ones(1, 1, 2, 2)
    logits = torch.full((1, 1, 1, 2, 2), 10.0)

    output = pool_entity_geometry(logits, points, depth)

    assert output["entity_geometry_valid"].item()
    assert torch.isfinite(output["entity_centers_world"]).all()
    assert torch.isfinite(output["entity_covariance_world"]).all()
    torch.testing.assert_close(
        output["entity_centers_world"],
        torch.ones(1, 1, 1, 3),
        atol=1e-4,
        rtol=1e-4,
    )
