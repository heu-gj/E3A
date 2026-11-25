import torch

import e3a.models.heads.geometry_head as geometry_head_module
from e3a.models.heads.geometry_head import GeometryHead


def test_geometry_head_exposes_camera_frame_entity_centers_and_covariance(monkeypatch):
    points_world = torch.tensor(
        [[[[[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
           [[0.0, 2.0, 1.0], [2.0, 2.0, 1.0]]]]]
    )
    extrinsics = torch.zeros(1, 1, 3, 4)
    extrinsics[..., :3, :3] = torch.eye(3)
    extrinsics[..., 0, 3] = 1.0
    extrinsics[..., 1, 3] = 2.0
    extrinsics[..., 2, 3] = 3.0

    def fake_decode_and_unproject(depth, pose_enc):
        return {
            "points_world": points_world,
            "extrinsics_camera_from_world": extrinsics,
            "intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
        }

    monkeypatch.setattr(
        geometry_head_module, "decode_and_unproject", fake_decode_and_unproject
    )
    output = GeometryHead()(
        depth=torch.ones(1, 1, 2, 2),
        pose_enc=torch.zeros(1, 1, 9),
        mask_logits=torch.full((1, 1, 1, 2, 2), 20.0),
    )

    torch.testing.assert_close(
        output["entity_centers_world"], torch.tensor([[[[1.0, 1.0, 1.0]]]])
    )
    torch.testing.assert_close(
        output["entity_centers_camera"], torch.tensor([[[[2.0, 3.0, 4.0]]]])
    )
    expected_covariance = torch.tensor(
        [[[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]]
    )
    torch.testing.assert_close(
        output["entity_covariance_camera"], expected_covariance
    )
