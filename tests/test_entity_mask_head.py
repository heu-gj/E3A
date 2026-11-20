import torch
from torch import nn

from e3a.models.heads.entity_mask_head import EntityMaskHead


class FakeMaskPredictor(nn.Module):
    def forward(self, queries, pixels):
        return torch.einsum("bqc,bchw->bqhw", queries, pixels)


def test_entity_mask_head_reuses_injected_predictor_contract():
    head = EntityMaskHead(
        entity_dim=32,
        pixel_dim=16,
        mask_dim=16,
        mask_predictor=FakeMaskPredictor(),
    )
    entities = torch.rand(2, 3, 4, 32, requires_grad=True)
    pixels = torch.rand(2, 3, 16, 8, 8, requires_grad=True)

    output = head(entities, pixels)

    assert output["mask_logits"].shape == (2, 3, 4, 8, 8)
    assert output["objectness_logits"].shape == (2, 3, 4)
    (output["mask_logits"].mean() + output["objectness_logits"].mean()).backward()
    assert entities.grad is not None
    assert pixels.grad is not None


def test_entity_mask_head_normalizes_bfloat16_backbone_features():
    head = EntityMaskHead(
        entity_dim=32,
        pixel_dim=16,
        mask_dim=16,
        mask_predictor=FakeMaskPredictor(),
    )
    entities = torch.rand(1, 2, 3, 32).to(torch.bfloat16)
    pixels = torch.rand(1, 2, 16, 8, 8).to(torch.bfloat16)

    output = head(entities, pixels)

    assert output["mask_logits"].dtype == torch.float32
    assert output["objectness_logits"].dtype == torch.float32
    assert torch.isfinite(output["mask_logits"]).all()
