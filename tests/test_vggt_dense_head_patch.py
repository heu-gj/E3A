from types import MethodType

import torch
from torch import nn

from vggt_omega.models.heads.dense_head import DenseHead


def make_lightweight_dense_head():
    head = DenseHead.__new__(DenseHead)
    nn.Module.__init__(head)
    calls = []

    def fake_forward_impl(
        self,
        aggregated_tokens_list,
        images,
        patch_token_start,
        frames_start_idx=None,
        frames_end_idx=None,
        return_features=False,
    ):
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx]
        batch, frames, _, height, width = images.shape
        marker = float(frames_start_idx or 0)
        depth = images.new_full((batch, frames, height, width, 1), marker + 1)
        confidence = images.new_full((batch, frames, height, width), marker + 2)
        fused = images.new_full((batch, frames, 8, height // 4, width // 4), marker + 3)
        calls.append((frames_start_idx, frames_end_idx, return_features))
        if return_features:
            return depth, confidence, fused
        return depth, confidence

    head._forward_impl = MethodType(fake_forward_impl, head)
    return head, calls


def test_dense_head_feature_return_preserves_frame_chunking():
    head, calls = make_lightweight_dense_head()
    images = torch.rand(1, 5, 3, 16, 16)

    depth, confidence, fused = head(
        [None],
        images,
        patch_token_start=2,
        frames_chunk_size=2,
        return_features=True,
    )

    assert depth.shape == (1, 5, 16, 16, 1)
    assert confidence.shape == (1, 5, 16, 16)
    assert fused.shape == (1, 5, 8, 4, 4)
    assert calls == [(0, 2, True), (2, 4, True), (4, 5, True)]


def test_dense_head_default_contract_remains_two_outputs():
    head, _ = make_lightweight_dense_head()
    images = torch.rand(1, 2, 3, 16, 16)

    output = head(
        [None],
        images,
        patch_token_start=2,
        frames_chunk_size=None,
    )

    assert len(output) == 2
