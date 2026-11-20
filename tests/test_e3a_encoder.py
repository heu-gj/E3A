import torch
from torch import nn

from e3a.models.encoder import E3AEncoder
from e3a.models.token_bank import E3ATokenBank


class FakeBackbone(nn.Module):
    def forward_with_features(self, images, *, extra_tokens):
        return {
            "depth": images.new_ones(images.shape[:2] + images.shape[-2:]),
            "fused_features": images.new_zeros(
                images.shape[0],
                images.shape[1],
                8,
                images.shape[-2] // 4,
                images.shape[-1] // 4,
            ),
            "extra_tokens": extra_tokens * 2,
        }


class FakeAggregator:
    patch_token_start = 17
    camera_token = torch.empty(1, 2, 1, 32)


class FakeModel:
    aggregator = FakeAggregator()


class FakeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeModel()
        self.fake = FakeBackbone()

    def forward_with_features(self, images, *, extra_tokens):
        return self.fake.forward_with_features(images, extra_tokens=extra_tokens)


def test_e3a_encoder_splits_entity_hand_future_tokens():
    bank = E3ATokenBank(embed_dim=32, num_entities=4, num_hands=2)
    encoder = E3AEncoder(FakeAdapter(), bank)

    images = torch.rand(2, 3, 3, 64, 64)
    timestamps = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])
    output = encoder(images, timestamps)

    assert output["entity_tokens"].shape == (2, 3, 4, 32)
    assert output["hand_tokens"].shape == (2, 3, 2, 32)
    assert output["future_tokens"].shape == (2, 3, 1, 32)
    assert output["depth"].shape == (2, 3, 64, 64)
