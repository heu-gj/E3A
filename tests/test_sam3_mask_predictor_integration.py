import sys
from pathlib import Path

import pytest
import torch

from e3a.models.heads.entity_mask_head import EntityMaskHead


def test_real_sam3_mask_predictor_contract_when_sam3_is_installed():
    sam3_root = Path(__file__).resolve().parents[1] / "third_party" / "sam3"
    if sam3_root.is_dir() and str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    pytest.importorskip("sam3.model.maskformer_segmentation")

    head = EntityMaskHead(
        entity_dim=8,
        pixel_dim=8,
        mask_dim=8,
    )
    entity_tokens = torch.randn(1, 2, 3, 8, requires_grad=True)
    fused_features = torch.randn(1, 2, 8, 4, 4, requires_grad=True)

    output = head(entity_tokens, fused_features)

    assert output["mask_logits"].shape == (1, 2, 3, 4, 4)
    assert output["objectness_logits"].shape == (1, 2, 3)
    assert torch.isfinite(output["mask_logits"]).all()

    (output["mask_logits"].mean() + output["objectness_logits"].mean()).backward()
    assert entity_tokens.grad is not None
    assert fused_features.grad is not None
