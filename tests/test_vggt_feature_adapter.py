import torch
from torch import nn

from e3a.backbones.vggt_omega import VGGTOmegaAdapter


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_token_start = 2
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, 4))

    def forward(self, images, extra_tokens=None):
        batch, frames = images.shape[:2]
        count = 0 if extra_tokens is None else extra_tokens.shape[2]
        prefix = self.patch_token_start + count
        final = images.new_zeros(batch, frames, prefix + 4, 8)
        if extra_tokens is not None:
            final[:, :, self.patch_token_start:prefix, :4] = extra_tokens
        return [final], prefix


class FakeCameraHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_prefix = None

    def forward(self, tokens, patch_token_start):
        self.seen_prefix = patch_token_start
        return tokens[-1].new_zeros(tokens[-1].shape[:2] + (9,))


class FakeDenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_prefix = None

    def forward(self, tokens, images, patch_token_start, return_features=False):
        self.seen_prefix = patch_token_start
        batch, frames, _, height, width = images.shape
        depth = images.new_ones(batch, frames, height, width)
        confidence = images.new_ones(batch, frames, height, width)
        fused = images.new_zeros(batch, frames, 8, height // 4, width // 4)
        if return_features:
            return depth, confidence, fused
        return depth, confidence


class FakeVGGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = FakeAggregator()
        self.camera_head = FakeCameraHead()
        self.dense_head = FakeDenseHead()
        self.text_alignment_head = None
        self.autocast = False


def test_feature_path_keeps_camera_prefix_separate_from_e3a_prefix():
    model = FakeVGGT()
    adapter = VGGTOmegaAdapter(model=model, freeze=False)
    images = torch.rand(1, 2, 3, 16, 16)
    extra_tokens = torch.rand(1, 2, 3, 4)

    output = adapter.forward_with_features(images, extra_tokens=extra_tokens)

    assert model.camera_head.seen_prefix == 2
    assert model.dense_head.seen_prefix == 5
    assert output["extra_tokens"].shape == (1, 2, 3, 8)
    assert output["fused_features"].shape == (1, 2, 8, 4, 4)
