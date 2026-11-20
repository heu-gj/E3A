import torch

from e3a.models.heads.entity_mask_head import EntityMaskHead


def test_native_entity_mask_head_predicts_masks_and_backpropagates():
    head = EntityMaskHead(
        entity_dim=32,
        pixel_dim=16,
        mask_dim=16,
    )
    entities = torch.randn(2, 3, 4, 32, requires_grad=True)
    pixels = torch.randn(2, 3, 16, 8, 8, requires_grad=True)

    output = head(entities, pixels)

    assert output["mask_logits"].shape == (2, 3, 4, 8, 8)
    assert output["objectness_logits"].shape == (2, 3, 4)
    assert torch.isfinite(output["mask_logits"]).all()
    (output["mask_logits"].mean() + output["objectness_logits"].mean()).backward()
    assert entities.grad is not None and torch.isfinite(entities.grad).all()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()


def test_native_entity_mask_head_accepts_bfloat16_backbone_features():
    head = EntityMaskHead(entity_dim=32, pixel_dim=16, mask_dim=16)
    entities = torch.randn(1, 2, 3, 32).to(torch.bfloat16)
    pixels = torch.randn(1, 2, 16, 8, 8).to(torch.bfloat16)

    output = head(entities, pixels)

    assert output["mask_logits"].dtype == torch.float32
    assert output["objectness_logits"].dtype == torch.float32
    assert torch.isfinite(output["mask_logits"]).all()
