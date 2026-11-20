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
    assert losses["loss_hand"].item() == 0
    assert losses["loss_interaction"].item() == 0
    assert losses["loss_target_entity"].item() == 0
    assert losses["loss_action"].item() == 0
    assert losses["loss_time"].item() == 0
