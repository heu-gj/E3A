import torch

from e3a.geometry.vggt_geometry import decode_and_unproject
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


def test_official_vggt_unprojection_identity_camera():
    extrinsics = torch.zeros(1, 1, 3, 4)
    extrinsics[..., :3, :3] = torch.eye(3)
    intrinsics = torch.zeros(1, 1, 3, 3)
    intrinsics[..., 0, 0] = 1.0
    intrinsics[..., 1, 1] = 1.0
    intrinsics[..., 0, 2] = 1.0
    intrinsics[..., 1, 2] = 1.0
    intrinsics[..., 2, 2] = 1.0

    pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, (2, 2))
    depth = torch.ones(1, 1, 2, 2)

    output = decode_and_unproject(depth, pose_enc)
    points = output["points_world"]

    expected = torch.tensor(
        [[
            [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0]],
            [[-0.5, 0.5, 1.0], [0.5, 0.5, 1.0]],
        ]]
    ).unsqueeze(0)

    torch.testing.assert_close(points, expected, atol=1e-5, rtol=1e-5)
