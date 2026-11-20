import torch
from torch import nn

from e3a.models.representation import E3ARepresentationModel
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        extrinsics = torch.zeros(1, 1, 3, 4)
        extrinsics[..., :3, :3] = torch.eye(3)
        intrinsics = torch.zeros(1, 1, 3, 3)
        intrinsics[..., 0, 0] = 1.0
        intrinsics[..., 1, 1] = 1.0
        intrinsics[..., 0, 2] = 1.0
        intrinsics[..., 1, 2] = 1.0
        intrinsics[..., 2, 2] = 1.0
        self.pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, (2, 2))

    def forward(self, images, timestamps):
        batch, frames = images.shape[:2]
        return {
            "entity_tokens": images.new_zeros(batch, frames, 1, 2048),
            "fused_features": images.new_zeros(batch, frames, 256, 1, 1),
            "depth": images.new_ones(batch, frames, 2, 2, 1),
            "pose_enc": self.pose_enc.expand(batch, frames, -1),
        }


class FakeMaskHead(nn.Module):
    def forward(self, entity_tokens, fused_features):
        batch, frames, entities = entity_tokens.shape[:3]
        return {
            "mask_logits": entity_tokens.new_full((batch, frames, entities, 2, 2), 10.0),
            "objectness_logits": entity_tokens.new_ones(batch, frames, entities),
        }


def test_representation_model_composes_mask_and_world_geometry():
    model = E3ARepresentationModel(
        encoder=FakeEncoder(),
        mask_head=FakeMaskHead(),
    )
    images = torch.rand(1, 1, 3, 2, 2)
    timestamps = torch.zeros(1, 1)

    output = model(images, timestamps)

    assert output["mask_logits"].shape == (1, 1, 1, 2, 2)
    assert output["points_world"].shape == (1, 1, 2, 2, 3)
    assert output["entity_centers_world"].shape == (1, 1, 1, 3)
    assert output["entity_geometry_valid"].item()
