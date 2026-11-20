import pytest
import torch
from torch import nn

from e3a.adapters.videomt import VidEoMTMaskCriterion


class RecordingMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.outputs = None
        self.targets = None

    def forward(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets
        return [
            (
                torch.tensor([0], dtype=torch.int64, device=outputs["pred_logits"].device),
                torch.tensor([0], dtype=torch.int64, device=outputs["pred_logits"].device),
            )
            if target["labels"].numel()
            else (
                torch.empty(0, dtype=torch.int64, device=outputs["pred_logits"].device),
                torch.empty(0, dtype=torch.int64, device=outputs["pred_logits"].device),
            )
            for target in targets
        ]


class RecordingCriterion(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher

    def forward(self, outputs, targets):
        self.matcher(outputs, targets)
        masks = outputs["pred_masks"]
        logits = outputs["pred_logits"]
        return {
            "loss_ce": logits.square().mean(),
            "loss_mask": masks.square().mean(),
            "loss_dice": masks.abs().mean(),
        }

    def loss_labels(self, outputs, targets, indices, num_masks):
        return {"loss_ce": outputs["pred_logits"].square().mean()}


def make_criterion(frames=2, matcher=None, criterion=None):
    matcher = matcher or RecordingMatcher()
    criterion = criterion or RecordingCriterion(matcher)
    return VidEoMTMaskCriterion(
        frames=frames,
        num_points=32,
        loss_weights={"loss_ce": 1.0, "loss_mask": 2.0, "loss_dice": 3.0},
        matcher=matcher,
        criterion=criterion,
    )


def make_clip_target():
    labels = torch.tensor([0], dtype=torch.int64)
    ids = torch.tensor([[41, 41]], dtype=torch.int64)
    masks = torch.zeros(1, 2, 8, 8)
    masks[0, 0, 1:4, 1:4] = 1
    masks[0, 1, 2:6, 2:6] = 1
    return {"labels": labels, "ids": ids, "masks": masks}


def test_adapter_converts_e3a_outputs_to_official_videomt_frame_contract():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [make_clip_target()],
    )

    assert matcher.outputs["pred_masks"].shape == (2, 3, 1, 8, 8)
    assert matcher.outputs["pred_logits"].shape == (2, 3, 2)
    torch.testing.assert_close(
        matcher.outputs["pred_logits"][0],
        matcher.outputs["pred_logits"][1],
    )
    assert len(matcher.targets) == 2
    assert matcher.targets[0]["ids"].shape == (1, 1)
    assert matcher.targets[1]["ids"].shape == (1, 1)
    assert matcher.targets[0]["masks"].shape == (1, 1, 8, 8)
    assert matcher.targets[1]["masks"].shape == (1, 1, 8, 8)
    torch.testing.assert_close(matcher.targets[0]["ids"], torch.tensor([[41]]))
    torch.testing.assert_close(matcher.targets[1]["ids"], torch.tensor([[41]]))

    expected_objective = (
        losses["loss_ce"] + 2.0 * losses["loss_mask"] + 3.0 * losses["loss_dice"]
    )
    torch.testing.assert_close(losses["loss_objective"], expected_objective)
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None


def test_adapter_handles_an_all_empty_batch_without_sampling_empty_masks():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)
    empty_target = {
        "labels": torch.empty(0, dtype=torch.int64),
        "ids": torch.empty(0, 2, dtype=torch.int64),
        "masks": torch.empty(0, 2, 8, 8),
    }

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [empty_target],
    )

    assert losses["loss_ce"].isfinite()
    assert losses["loss_mask"].item() == 0
    assert losses["loss_dice"].item() == 0
    losses["loss_objective"].backward()
    assert objectness_logits.grad is not None
    assert mask_logits.grad is not None
    assert torch.count_nonzero(mask_logits.grad) == 0


def test_adapter_rejects_track_id_changes_within_a_clip():
    adapter = make_criterion()
    target = make_clip_target()
    target["ids"] = torch.tensor([[41, 42]], dtype=torch.int64)

    with pytest.raises(ValueError, match="changes ID across frames"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            [target],
        )


def test_adapter_rejects_more_visible_tracks_than_entity_slots():
    adapter = make_criterion()
    target = {
        "labels": torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        "ids": torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.int64),
        "masks": torch.ones(4, 2, 8, 8),
    }

    with pytest.raises(ValueError, match="only 3 entity slots"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            [target],
        )


def test_adapter_rejects_nonzero_mask_for_a_missing_track_frame():
    adapter = make_criterion()
    target = make_clip_target()
    target["ids"] = torch.tensor([[41, -1]], dtype=torch.int64)
    target["masks"][0, 1, 2:6, 2:6] = 0.5

    with pytest.raises(ValueError, match="zero masks wherever track ID is -1"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            [target],
        )


def test_real_videomt_criterion_contract_when_optional_dependencies_are_installed():
    pytest.importorskip("detectron2")
    pytest.importorskip("scipy")

    adapter = VidEoMTMaskCriterion(
        frames=2,
        num_points=32,
        loss_weights={"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0},
    )
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [make_clip_target()],
    )

    assert {"loss_ce", "loss_mask", "loss_dice", "loss_objective"} <= set(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None


def test_adapter_accepts_the_official_vggt_trainer_batch_signature():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)
    clip_target = make_clip_target()
    batch = {
        "images": torch.rand(1, 2, 3, 8, 8),
        "timestamps": torch.tensor([[0.0, 0.1]]),
        "entity_targets": {
            "labels": torch.tensor([[0]], dtype=torch.int64),
            "ids": clip_target["ids"].unsqueeze(0),
            "masks": clip_target["masks"].unsqueeze(0),
            "track_counts": torch.tensor([1], dtype=torch.int64),
        },
    }

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        batch,
        schedule_progress=0.25,
    )

    assert torch.isfinite(losses["loss_objective"])
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None


def test_padded_targets_stay_batch_aligned_under_upstream_gradient_accumulation():
    from pathlib import Path
    import sys

    training_root = Path(__file__).resolve().parents[1] / "third_party" / "vggt-omega" / "training"
    if not training_root.is_dir():
        pytest.skip("pinned VGGT-Ω training utilities are not checked out")
    sys.path.insert(0, str(training_root))
    from trainer_utils import get_chunk_from_data

    batch_size, max_tracks, frames = 4, 2, 2
    labels = torch.zeros(batch_size, max_tracks, dtype=torch.int64)
    ids = torch.full((batch_size, max_tracks, frames), -1, dtype=torch.int64)
    masks = torch.zeros(batch_size, max_tracks, frames, 8, 8)
    counts = torch.tensor([1, 0, 2, 1], dtype=torch.int64)
    for batch_index, count in enumerate(counts.tolist()):
        for track_index in range(count):
            ids[batch_index, track_index] = 100 + batch_index * 10 + track_index
            masks[batch_index, track_index, 0, 1:4, 1:4] = 1
            masks[batch_index, track_index, 1, 2:6, 2:6] = 1

    batch = {
        "images": torch.rand(batch_size, frames, 3, 8, 8),
        "timestamps": torch.arange(frames).float().expand(batch_size, -1),
        "entity_targets": {
            "labels": labels,
            "ids": ids,
            "masks": masks,
            "track_counts": counts,
        },
    }
    chunk = get_chunk_from_data(batch, chunk_id=0, num_chunks=2)

    assert chunk["images"].shape[0] == 2
    assert chunk["entity_targets"]["labels"].shape[0] == 2
    assert chunk["entity_targets"]["ids"].shape[0] == 2
    assert chunk["entity_targets"]["masks"].shape[0] == 2
    assert chunk["entity_targets"]["track_counts"].tolist() == [1, 0]

    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    losses = adapter(
        {
            "mask_logits": torch.randn(2, frames, 3, 8, 8, requires_grad=True),
            "objectness_logits": torch.randn(2, frames, 3, requires_grad=True),
        },
        chunk,
        schedule_progress=0.5,
    )
    assert torch.isfinite(losses["loss_objective"])
    assert len(matcher.targets) == 2 * frames
    assert [target["labels"].numel() for target in matcher.targets] == [1, 1, 0, 0]


def test_adapter_rejects_trainer_batch_without_entity_targets():
    adapter = make_criterion()
    with pytest.raises(ValueError, match="entity_targets"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            {"images": torch.rand(1, 2, 3, 8, 8)},
            schedule_progress=0.0,
        )
