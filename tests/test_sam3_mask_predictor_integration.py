import pytest
import torch

from e3a.models.heads.entity_mask_head import EntityMaskHead


def test_real_sam3_mask_predictor_contract_when_sam3_is_installed():
    pytest.importorskip("sam3.model.maskformer_segmentation")

    head = EntityMaskHead(
        entity_dim=8,
        pixel_dim=8,
        mask_dim=8,
    )
    entity_tokens = torch.randn(1, 2, 3, 8)
    fused_features = torch.randn(1, 2, 8, 4, 4)

    output = head(entity_tokens, fused_features)

    assert output["mask_logits"].shape == (1, 2, 3, 4, 4)
    assert output["objectness_logits"].shape == (1, 2, 3)
    assert torch.isfinite(output["mask_logits"]).all()
