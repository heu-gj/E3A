"""Regression guard: SAM 3 remains optional and is not loaded by E3A's online mask head."""

import sys

from e3a.models.heads.entity_mask_head import EntityMaskHead


def test_native_entity_mask_head_does_not_import_sam3():
    head = EntityMaskHead(entity_dim=32, pixel_dim=16, mask_dim=16)
    assert head.__class__.__module__ == "e3a.models.heads.entity_mask_head"
    assert "sam3.model.maskformer_segmentation" not in sys.modules
