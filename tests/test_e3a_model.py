from types import SimpleNamespace

import torch
from torch import nn

from e3a.models.e3a_model import E3AModel


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_bank = SimpleNamespace(embed_dim=32)
        self.entity = nn.Parameter(torch.randn(1, 1, 3, 32))
        self.hand = nn.Parameter(torch.randn(1, 1, 2, 32))
        self.future = nn.Parameter(torch.randn(1, 1, 1, 32))
        self.pixels = nn.Parameter(torch.randn(1, 1, 8, 4, 4))

    def forward(self, images, timestamps):
        batch, frames = images.shape[:2]
        return {
            "entity_tokens": self.entity.expand(batch, frames, -1, -1),
            "hand_tokens": self.hand.expand(batch, frames, -1, -1),
            "future_tokens": self.future.expand(batch, frames, -1, -1),
            "fused_features": self.pixels.expand(batch, frames, -1, -1, -1),
            "depth": images.new_ones(batch, frames, 4, 4, 1),
            "pose_enc": images.new_zeros(batch, frames, 9),
        }


class FakeGeometryHead(nn.Module):
    def forward(self, *, depth, pose_enc, mask_logits):
        batch, frames, slots = mask_logits.shape[:3]
        extrinsics = depth.new_zeros(batch, frames, 3, 4)
        extrinsics[..., :3, :3] = torch.eye(3, device=depth.device)
        return {
            "points_world": depth.new_zeros(batch, frames, 4, 4, 3),
            "extrinsics_camera_from_world": extrinsics,
            "intrinsics": depth.new_zeros(batch, frames, 3, 3),
            "entity_centers_world": depth.new_zeros(batch, frames, slots, 3),
            "entity_covariance_world": depth.new_zeros(batch, frames, slots, 3, 3),
            "entity_geometry_valid": torch.ones(
                batch, frames, slots, dtype=torch.bool, device=depth.device
            ),
            "entity_mask_mass": depth.new_ones(batch, frames, slots),
        }


def test_e3a_model_emits_all_tasks_in_one_forward_and_backpropagates():
    model = E3AModel(
        FakeEncoder(),
        num_actions=7,
        pixel_dim=8,
        mask_dim=16,
        interaction_dim=16,
        geometry_head=FakeGeometryHead(),
    )
    images = torch.rand(2, 3, 3, 16, 16)
    timestamps = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])

    output = model(images, timestamps)

    assert output["mask_logits"].shape == (2, 3, 3, 4, 4)
    assert output["persistent_entity_tokens"].shape == (2, 3, 32)
    assert output["hand_keypoints_camera"].shape == (2, 3, 2, 21, 3)
    assert output["hand_keypoints_world"].shape == (2, 3, 2, 21, 3)
    assert output["interaction_logits"].shape == (2, 3, 2, 3)
    assert output["target_entity_logits"].shape == (2, 4)
    assert output["action_logits"].shape == (2, 7)
    assert output["time_log_mean"].shape == (2,)
    objective = (
        output["mask_logits"].mean()
        + output["interaction_logits"].mean()
        + output["target_entity_logits"].mean()
        + output["action_logits"].mean()
        + output["time_log_mean"].mean()
    )
    objective.backward()
    assert model.encoder.entity.grad is not None
    assert model.encoder.hand.grad is not None
    assert model.encoder.future.grad is not None
    assert model.encoder.pixels.grad is not None
