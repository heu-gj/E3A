import torch
from torch import nn

from vggt_omega.models.aggregator import Aggregator


class FakePatchEmbed(nn.Module):
    def forward(self, images):
        batch = images.shape[0]
        return images.new_zeros(batch, 1, 4)


class FakeRope(nn.Module):
    def forward(self, H, W):
        value = torch.zeros(1, 1, 1)
        return value, value


class AddOneBlock(nn.Module):
    def forward(self, tokens, rope_sincos):
        return tokens + 1


def make_lightweight_aggregator():
    aggregator = Aggregator.__new__(Aggregator)
    nn.Module.__init__(aggregator)
    aggregator.patch_embed = FakePatchEmbed()
    aggregator.rope_embed = FakeRope()
    aggregator.frame_blocks = nn.ModuleList([AddOneBlock()])
    aggregator.inter_frame_blocks = nn.ModuleList([AddOneBlock()])
    aggregator.depth = 1
    aggregator.patch_size = 16
    aggregator.use_checkpoint = False
    aggregator.cached_layer_indices = {0}
    aggregator.camera_token = nn.Parameter(torch.zeros(1, 2, 1, 4))
    aggregator.register_token = nn.Parameter(torch.zeros(1, 2, 2, 4))
    aggregator.patch_token_start = 3
    aggregator.inter_frame_attention_types = ["register"]
    aggregator.register_buffer("_resnet_mean", torch.zeros(1, 1, 3, 1, 1))
    aggregator.register_buffer("_resnet_std", torch.ones(1, 1, 3, 1, 1))
    return aggregator


def test_extra_tokens_join_register_attention_without_mutating_original_prefix():
    aggregator = make_lightweight_aggregator()
    images = torch.zeros(1, 2, 3, 16, 16)
    extra_tokens = torch.full((1, 2, 2, 4), 5.0, requires_grad=True)

    outputs, expanded_prefix = aggregator(images, extra_tokens=extra_tokens)
    cached = outputs[0]

    assert expanded_prefix == 5
    assert cached.shape == (1, 2, 6, 8)
    assert aggregator.patch_token_start == 3

    # The first half is the per-frame output (+1). The second half is the
    # inter-frame output: prefix tokens receive another +1, patch tokens do not.
    torch.testing.assert_close(cached[:, :, 3:5, :4], torch.full((1, 2, 2, 4), 6.0))
    torch.testing.assert_close(cached[:, :, 3:5, 4:], torch.full((1, 2, 2, 4), 7.0))
    torch.testing.assert_close(cached[:, :, 5:, 4:], torch.ones((1, 2, 1, 4)))

    cached[:, :, 3:5].sum().backward()
    assert extra_tokens.grad is not None
    torch.testing.assert_close(extra_tokens.grad, torch.full_like(extra_tokens, 2.0))


def test_no_extra_tokens_preserves_original_prefix_length():
    aggregator = make_lightweight_aggregator()
    images = torch.zeros(1, 2, 3, 16, 16)

    outputs, prefix = aggregator(images)

    assert prefix == 3
    assert outputs[0].shape == (1, 2, 4, 8)
    assert aggregator.patch_token_start == 3
