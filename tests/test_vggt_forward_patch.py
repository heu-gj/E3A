import torch
from torch import nn

from vggt_omega.models.vggt_omega import VGGTOmega


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_token_start = 2
        self.received_extra_tokens = None

    def forward(self, images, extra_tokens=None):
        self.received_extra_tokens = extra_tokens
        batch, frames = images.shape[:2]
        num_extra = 0 if extra_tokens is None else extra_tokens.shape[2]
        expanded_prefix = self.patch_token_start + num_extra
        tokens = images.new_zeros(batch, frames, expanded_prefix + 4, 8)
        if extra_tokens is not None:
            tokens[:, :, self.patch_token_start:expanded_prefix, :4] = extra_tokens
            tokens[:, :, self.patch_token_start:expanded_prefix, 4:] = extra_tokens * 2
        return [tokens], expanded_prefix


class FakeCameraHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_prefix = None

    def forward(self, aggregated_tokens_list, patch_token_start):
        self.seen_prefix = patch_token_start
        tokens = aggregated_tokens_list[-1]
        return tokens.new_zeros(tokens.shape[:2] + (9,))


class FakeDenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_prefix = None
        self.seen_return_features = None

    def forward(
        self,
        aggregated_tokens_list,
        images,
        patch_token_start,
        return_features=False,
    ):
        self.seen_prefix = patch_token_start
        self.seen_return_features = return_features
        batch, frames, _, height, width = images.shape
        depth = images.new_ones(batch, frames, height, width, 1)
        depth_conf = images.new_ones(batch, frames, height, width)
        if return_features:
            fused = images.new_ones(batch, frames, 256, height // 4, width // 4)
            return depth, depth_conf, fused
        return depth, depth_conf


def make_model():
    model = VGGTOmega.__new__(VGGTOmega)
    nn.Module.__init__(model)
    model.autocast = False
    model.aggregator = FakeAggregator()
    model.camera_head = FakeCameraHead()
    model.dense_head = FakeDenseHead()
    model.text_alignment_head = None
    return model


def original_unpatched_forward_reference(model, images):
    """Test-only reference copied from the pinned upstream forward contract."""
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    aggregated_tokens_list, patch_token_start = model.aggregator(images)
    final_tokens = aggregated_tokens_list[-1]
    predictions = {
        "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
    }
    with torch.autocast(device_type="cuda", enabled=False):
        if model.camera_head is not None:
            predictions["pose_enc"] = model.camera_head(
                aggregated_tokens_list,
                patch_token_start=patch_token_start,
            )
        if model.dense_head is not None:
            depth, depth_conf = model.dense_head(
                aggregated_tokens_list,
                images=images,
                patch_token_start=patch_token_start,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
        if model.text_alignment_head is not None:
            predictions.update(
                model.text_alignment_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )
            )
    if not model.training:
        predictions["images"] = images
    return predictions


def test_default_forward_matches_pinned_upstream_forward_contract():
    model = make_model().train()
    images = torch.rand(1, 2, 3, 32, 32)

    expected = original_unpatched_forward_reference(model, images)
    actual = model(images)

    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    assert model.aggregator.patch_token_start == 2
    assert model.camera_head.seen_prefix == 2
    assert model.dense_head.seen_prefix == 2
    assert model.dense_head.seen_return_features is False


def test_extra_tokens_are_exposed_without_changing_camera_prefix():
    model = make_model().eval()
    images = torch.rand(1, 2, 3, 32, 32)
    extra_tokens = torch.rand(1, 2, 3, 4)

    output = model(images, extra_tokens=extra_tokens, return_features=True)

    assert output["camera_and_register_tokens"].shape == (1, 2, 2, 8)
    assert output["extra_tokens"].shape == (1, 2, 3, 8)
    assert output["fused_features"].shape == (1, 2, 256, 8, 8)
    assert output["images"].shape == images.shape
    assert model.camera_head.seen_prefix == 2
    assert model.dense_head.seen_prefix == 5
    assert model.dense_head.seen_return_features is True
    assert model.aggregator.patch_token_start == 2


def test_return_features_requires_enabled_dense_head():
    model = make_model()
    model.dense_head = None
    try:
        model(torch.rand(1, 2, 3, 32, 32), return_features=True)
    except ValueError as exc:
        assert "DenseHead" in str(exc)
    else:
        raise AssertionError("Expected return_features without DenseHead to fail.")
