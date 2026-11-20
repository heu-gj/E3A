import torch
from torch import nn

from e3a.backbones.vggt_omega import VGGTOmegaAdapter


class FakePatchedVGGT(nn.Module):
    def forward(self, images, *, extra_tokens=None, return_features=False):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, frames, _, height, width = images.shape
        output = {
            "camera_and_register_tokens": images.new_zeros(batch, frames, 2, 8),
            "pose_enc": images.new_zeros(batch, frames, 9),
            "depth": images.new_ones(batch, frames, height, width, 1),
            "depth_conf": images.new_ones(batch, frames, height, width),
        }
        if extra_tokens is not None:
            output["extra_tokens"] = torch.cat([extra_tokens, extra_tokens * 2], dim=-1)
        if return_features:
            output["fused_features"] = images.new_zeros(
                batch, frames, 256, height // 4, width // 4
            )
        if not self.training:
            output["images"] = images
        return output


def test_feature_adapter_delegates_to_patched_official_forward():
    model = FakePatchedVGGT()
    adapter = VGGTOmegaAdapter(model=model, freeze=False)
    images = torch.rand(1, 2, 3, 16, 16)
    extra_tokens = torch.rand(1, 2, 3, 4)

    output = adapter.forward_with_features(images, extra_tokens=extra_tokens)

    assert output["camera_and_register_tokens"].shape == (1, 2, 2, 8)
    assert output["extra_tokens"].shape == (1, 2, 3, 8)
    assert output["fused_features"].shape == (1, 2, 256, 4, 4)
    assert output["depth"].shape == (1, 2, 16, 16, 1)


def test_feature_adapter_rejects_unpatched_upstream_model():
    class UnpatchedVGGT(nn.Module):
        def forward(self, images):
            return {"depth": images.new_ones(1)}

    adapter = VGGTOmegaAdapter(model=UnpatchedVGGT(), freeze=False)
    try:
        adapter.forward_with_features(
            torch.rand(1, 2, 3, 16, 16),
            extra_tokens=torch.rand(1, 2, 3, 4),
        )
    except RuntimeError as exc:
        assert "feature API patch is not applied" in str(exc)
    else:
        raise AssertionError("Expected an unpatched upstream model to fail closed.")
