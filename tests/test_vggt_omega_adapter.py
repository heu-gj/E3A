import torch
from torch import nn

from e3a.backbones.vggt_omega import VGGTOmegaAdapter


class FakeVGGT(nn.Module):
    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        return {
            "images": images,
            "depth": images.new_ones(images.shape[:2] + images.shape[-2:]),
        }


def test_adapter_passes_official_outputs_through():
    adapter = VGGTOmegaAdapter(model=FakeVGGT(), freeze=False)

    images = torch.rand(2, 3, 32, 32)
    output = adapter(images)

    assert set(output) == {"images", "depth"}
    assert output["images"].shape == (1, 2, 3, 32, 32)
    assert output["depth"].shape == (1, 2, 32, 32)


def test_adapter_accepts_batched_video():
    adapter = VGGTOmegaAdapter(model=FakeVGGT(), freeze=False)

    images = torch.rand(2, 4, 3, 32, 32)
    output = adapter(images)

    assert output["images"].shape == (2, 4, 3, 32, 32)


def test_adapter_rejects_invalid_input_layout():
    adapter = VGGTOmegaAdapter(model=FakeVGGT(), freeze=False)

    try:
        adapter(torch.rand(2, 32, 32))
    except ValueError as exc:
        assert "expects" in str(exc)
    else:
        raise AssertionError("Expected invalid input layout to fail.")
