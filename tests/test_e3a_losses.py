import pytest
import torch
from torch import nn

from e3a.engine.losses import E3AMultiTaskLoss


class FakeMaskCriterion(nn.Module):
    def forward_with_matches(self, predictions, batch, schedule_progress=None):
        loss = predictions["mask_logits"].square().mean()
        return {"loss_objective": loss}, [{101: 1}]


def test_joint_e3a_loss_maps_gt_track_to_entity_slot_and_backpropagates():
    predictions = {
        "mask_logits": torch.randn(1, 2, 3, 4, 4, requires_grad=True),
        "hand_keypoints_camera": torch.randn(1, 2, 2, 21, 3, requires_grad=True),
        "hand_joint_visibility_logits": torch.randn(1, 2, 2, 21, requires_grad=True),
        "interaction_logits": torch.randn(1, 2, 2, 3, requires_grad=True),
        "target_entity_logits": torch.randn(1, 4, requires_grad=True),
        "action_logits": torch.randn(1, 5, requires_grad=True),
        "time_log_mean": torch.randn(1, requires_grad=True),
        "time_log_scale": torch.zeros(1, requires_grad=True),
    }
    batch = {
        "entity_targets": {
            "ids": torch.tensor([[[101, 101]]]),
            "track_counts": torch.tensor([1]),
        },
        "hand_targets": {
            "keypoints_camera": torch.zeros(1, 2, 2, 21, 3),
            "valid": torch.zeros(1, 2, 2, 21, dtype=torch.bool),
            "visibility": torch.zeros(1, 2, 2, 21),
            "visibility_valid": torch.zeros(1, 2, 2, 21, dtype=torch.bool),
        },
        "interaction_targets": {
            "contact": torch.zeros(1, 2, 2, 1),
            "valid": torch.zeros(1, 2, 2, 1, dtype=torch.bool),
        },
        "anticipation_targets": {
            "target_entity_track_ids": torch.tensor([101]),
            "target_entity_valid": torch.tensor([True]),
            "action_labels": torch.tensor([2]),
            "action_valid": torch.tensor([True]),
            "time_to_event": torch.tensor([0.8]),
            "time_valid": torch.tensor([True]),
        },
    }
    batch["hand_targets"]["valid"][0, 0, 0, 0] = True
    batch["interaction_targets"]["contact"][0, 0, 0, 0] = 1.0
    batch["interaction_targets"]["valid"][0, 0, 0, 0] = True

    criterion = E3AMultiTaskLoss(FakeMaskCriterion())
    losses = criterion(predictions, batch, schedule_progress=0.5)

    assert set(losses) == {
        "loss_entity",
        "loss_geometry_center",
        "loss_geometry_covariance",
        "loss_geometry",
        "loss_hand",
        "loss_hand_visibility",
        "loss_interaction",
        "loss_target_entity",
        "loss_action",
        "loss_time",
        "loss_objective",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss_objective"].backward()
    for key in (
        "mask_logits",
        "hand_keypoints_camera",
        "interaction_logits",
        "target_entity_logits",
        "action_logits",
        "time_log_mean",
        "time_log_scale",
    ):
        assert predictions[key].grad is not None


def test_joint_e3a_loss_ignores_unannotated_tasks_instead_of_making_negatives():
    predictions = {
        "mask_logits": torch.randn(1, 1, 2, 4, 4, requires_grad=True),
        "hand_keypoints_camera": torch.randn(1, 1, 2, 21, 3, requires_grad=True),
        "hand_joint_visibility_logits": torch.randn(1, 1, 2, 21, requires_grad=True),
        "interaction_logits": torch.randn(1, 1, 2, 2, requires_grad=True),
        "target_entity_logits": torch.randn(1, 3, requires_grad=True),
        "action_logits": torch.randn(1, 4, requires_grad=True),
        "time_log_mean": torch.randn(1, requires_grad=True),
        "time_log_scale": torch.zeros(1, requires_grad=True),
    }
    batch = {
        "entity_targets": {
            "ids": torch.empty(1, 0, 1, dtype=torch.int64),
            "track_counts": torch.tensor([0]),
        },
        "anticipation_targets": {
            "target_entity_track_ids": torch.tensor([-1]),
            "target_entity_valid": torch.tensor([False]),
            "action_labels": torch.tensor([-1]),
            "action_valid": torch.tensor([False]),
            "time_to_event": torch.tensor([0.0]),
            "time_valid": torch.tensor([False]),
        },
    }
    losses = E3AMultiTaskLoss(FakeMaskCriterion())(predictions, batch)
    assert losses["loss_geometry"].item() == 0
    assert losses["loss_hand"].item() == 0
    assert losses["loss_interaction"].item() == 0
    assert losses["loss_target_entity"].item() == 0
    assert losses["loss_action"].item() == 0
    assert losses["loss_time"].item() == 0



def test_hand_loss_ignores_nonfinite_values_only_where_validity_is_false():
    criterion = E3AMultiTaskLoss(FakeMaskCriterion())
    predictions = {
        "hand_keypoints_camera": torch.zeros(1, 1, 2, 21, 3, requires_grad=True),
        "hand_joint_visibility_logits": torch.zeros(1, 1, 2, 21, requires_grad=True),
    }
    keypoints = torch.full((1, 1, 2, 21, 3), float("nan"))
    valid = torch.zeros(1, 1, 2, 21, dtype=torch.bool)
    visibility = torch.full((1, 1, 2, 21), float("nan"))
    visibility_valid = torch.zeros(1, 1, 2, 21, dtype=torch.bool)

    keypoints[0, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    valid[0, 0, 0, 0] = True
    visibility[0, 0, 0, 0] = 1.0
    visibility_valid[0, 0, 0, 0] = True

    hand_loss, visibility_loss = criterion._hand_losses(
        predictions,
        {
            "hand_targets": {
                "keypoints_camera": keypoints,
                "valid": valid,
                "visibility": visibility,
                "visibility_valid": visibility_valid,
            }
        },
    )

    assert torch.isfinite(hand_loss)
    assert torch.isfinite(visibility_loss)
    (hand_loss + visibility_loss).backward()
    assert predictions["hand_keypoints_camera"].grad is not None
    assert predictions["hand_joint_visibility_logits"].grad is not None


def test_hand_loss_rejects_nonfinite_values_at_valid_joint_locations():
    criterion = E3AMultiTaskLoss(FakeMaskCriterion())
    predictions = {
        "hand_keypoints_camera": torch.zeros(1, 1, 2, 21, 3),
        "hand_joint_visibility_logits": torch.zeros(1, 1, 2, 21),
    }
    keypoints = torch.zeros(1, 1, 2, 21, 3)
    keypoints[0, 0, 0, 0, 0] = float("nan")
    valid = torch.zeros(1, 1, 2, 21, dtype=torch.bool)
    valid[0, 0, 0, 0] = True

    with pytest.raises(ValueError, match="Valid 3D hand keypoints must be finite"):
        criterion._hand_losses(
            predictions,
            {"hand_targets": {"keypoints_camera": keypoints, "valid": valid}},
        )


def test_anticipation_rejects_invalid_negative_target_entity_ids():
    criterion = E3AMultiTaskLoss(FakeMaskCriterion())
    predictions = {
        "target_entity_logits": torch.zeros(1, 4),
        "action_logits": torch.zeros(1, 3),
        "time_log_mean": torch.zeros(1),
        "time_log_scale": torch.zeros(1),
    }
    targets = {
        "target_entity_track_ids": torch.tensor([-2]),
        "target_entity_valid": torch.tensor([True]),
        "action_labels": torch.tensor([-1]),
        "action_valid": torch.tensor([False]),
        "time_to_event": torch.tensor([0.0]),
        "time_valid": torch.tensor([False]),
    }

    with pytest.raises(ValueError, match="must be -1"):
        criterion._anticipation_losses(predictions, {"anticipation_targets": targets}, [{}])



def test_geometry_loss_reuses_clip_match_and_is_scale_invariant():
    criterion = E3AMultiTaskLoss(FakeMaskCriterion())
    target_centers = torch.tensor(
        [[[[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]]]]
    )
    target_covariance = torch.tensor(
        [[[[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]],
           [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]]]]]
    )
    predicted_centers = torch.zeros(1, 2, 3, 3)
    predicted_centers[:, :, 1] = target_centers.permute(0, 2, 1, 3).squeeze(2) * 2.0
    predicted_centers.requires_grad_()
    predicted_covariance = torch.zeros(1, 2, 3, 3, 3)
    predicted_covariance[:, :, 1] = (
        target_covariance.permute(0, 2, 1, 3, 4).squeeze(2) * 4.0
    )
    predicted_covariance.requires_grad_()
    predictions = {
        "mask_logits": torch.zeros(1, 2, 3, 2, 2, requires_grad=True),
        "entity_centers_camera": predicted_centers,
        "entity_covariance_camera": predicted_covariance,
        "entity_geometry_valid": torch.ones(1, 2, 3, dtype=torch.bool),
    }
    batch = {
        "entity_targets": {
            "ids": torch.tensor([[[101, 101]]]),
            "track_counts": torch.tensor([1]),
        },
        "geometry_targets": {
            "centers_camera": target_centers,
            "covariance_camera": target_covariance,
            "valid": torch.ones(1, 1, 2, dtype=torch.bool),
            "metric_valid": torch.tensor([False]),
            "units": ["unknown"],
            "normalization": "scale_invariant_per_clip",
        },
    }

    center_loss, covariance_loss, geometry_loss = criterion._geometry_losses(
        predictions, batch, [{101: 1}]
    )
    assert center_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert covariance_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert geometry_loss.item() == pytest.approx(0.0, abs=1e-7)

    with torch.no_grad():
        predicted_centers[0, 0, 1, 0] += 0.25
    center_loss, covariance_loss, geometry_loss = criterion._geometry_losses(
        predictions, batch, [{101: 1}]
    )
    assert center_loss.item() > 0
    assert torch.isfinite(covariance_loss)
    geometry_loss.backward()
    assert predicted_centers.grad is not None
    assert predicted_centers.grad.abs().sum().item() > 0
    assert predicted_covariance.grad is not None


def test_geometry_loss_ignores_invalid_geometry_frames():
    criterion = E3AMultiTaskLoss(FakeMaskCriterion())
    predictions = {
        "mask_logits": torch.zeros(1, 2, 2, 2, 2),
        "entity_centers_camera": torch.full((1, 2, 2, 3), float("nan")),
        "entity_covariance_camera": torch.full((1, 2, 2, 3, 3), float("nan")),
        "entity_geometry_valid": torch.zeros(1, 2, 2, dtype=torch.bool),
    }
    batch = {
        "entity_targets": {
            "ids": torch.tensor([[[101, 101]]]),
            "track_counts": torch.tensor([1]),
        },
        "geometry_targets": {
            "centers_camera": torch.full((1, 1, 2, 3), float("nan")),
            "covariance_camera": torch.full((1, 1, 2, 3, 3), float("nan")),
            "valid": torch.zeros(1, 1, 2, dtype=torch.bool),
        },
    }
    center_loss, covariance_loss, geometry_loss = criterion._geometry_losses(
        predictions, batch, [{101: 1}]
    )
    assert center_loss.item() == 0
    assert covariance_loss.item() == 0
    assert geometry_loss.item() == 0


def test_geometry_loss_rejects_track_id_changes_within_a_clip():
    criterion = E3AMultiTaskLoss(FakeMaskCriterion())
    predictions = {
        "mask_logits": torch.zeros(1, 2, 2, 2, 2),
        "entity_centers_camera": torch.ones(1, 2, 2, 3),
        "entity_covariance_camera": torch.ones(1, 2, 2, 3, 3),
        "entity_geometry_valid": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    batch = {
        "entity_targets": {
            "ids": torch.tensor([[[101, 102]]]),
            "track_counts": torch.tensor([1]),
        },
        "geometry_targets": {
            "centers_camera": torch.ones(1, 1, 2, 3),
            "covariance_camera": torch.ones(1, 1, 2, 3, 3),
            "valid": torch.ones(1, 1, 2, dtype=torch.bool),
        },
    }
    with pytest.raises(ValueError, match="one clip-local track ID"):
        criterion._geometry_losses(predictions, batch, [{101: 1}])



def test_geometry_weight_is_included_in_joint_objective():
    criterion = E3AMultiTaskLoss(
        FakeMaskCriterion(),
        weights={
            "entity": 0.0,
            "geometry": 2.0,
            "hand": 0.0,
            "hand_visibility": 0.0,
            "interaction": 0.0,
            "target_entity": 0.0,
            "action": 0.0,
            "time": 0.0,
        },
    )
    target_centers = torch.tensor(
        [[[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]]]
    )
    predicted_centers = torch.zeros(1, 2, 3, 3)
    predicted_centers[:, :, 1] = (
        target_centers.permute(0, 2, 1, 3).squeeze(2) + torch.tensor([0.5, 0.0, 0.0])
    )
    predictions = {
        "mask_logits": torch.zeros(1, 2, 3, 2, 2),
        "entity_centers_camera": predicted_centers,
        "entity_covariance_camera": torch.zeros(1, 2, 3, 3, 3),
        "entity_geometry_valid": torch.ones(1, 2, 3, dtype=torch.bool),
        "hand_keypoints_camera": torch.zeros(1, 2, 2, 21, 3),
        "hand_joint_visibility_logits": torch.zeros(1, 2, 2, 21),
        "interaction_logits": torch.zeros(1, 2, 2, 3),
        "target_entity_logits": torch.zeros(1, 4),
        "action_logits": torch.zeros(1, 2),
        "time_log_mean": torch.zeros(1),
        "time_log_scale": torch.zeros(1),
    }
    batch = {
        "entity_targets": {
            "ids": torch.tensor([[[101, 101]]]),
            "track_counts": torch.tensor([1]),
        },
        "geometry_targets": {
            "centers_camera": target_centers,
            "covariance_camera": torch.zeros(1, 1, 2, 3, 3),
            "valid": torch.ones(1, 1, 2, dtype=torch.bool),
        },
    }

    losses = criterion(predictions, batch)
    assert losses["loss_geometry"].item() > 0
    torch.testing.assert_close(
        losses["loss_objective"], 2.0 * losses["loss_geometry"]
    )
