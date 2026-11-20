import pytest
import torch

from e3a.data.collate import collate_entity_targets


def make_clip_target(track_count: int, *, frames: int = 2, size: int = 8):
    labels = torch.zeros(track_count, dtype=torch.int64)
    ids = torch.full((track_count, frames), -1, dtype=torch.int64)
    masks = torch.zeros(track_count, frames, size, size, dtype=torch.bool)
    for track in range(track_count):
        ids[track, :] = 100 + track
        masks[track, 0, 1:4, 1:4] = True
        masks[track, 1, 2:6, 2:6] = True
    return {"labels": labels, "ids": ids, "masks": masks}


def test_collator_pads_variable_track_counts_for_batch_first_chunking():
    targets = [
        make_clip_target(2),
        make_clip_target(0),
        make_clip_target(1),
    ]

    batch = collate_entity_targets(targets)

    assert batch["labels"].shape == (3, 2)
    assert batch["ids"].shape == (3, 2, 2)
    assert batch["masks"].shape == (3, 2, 2, 8, 8)
    assert batch["track_counts"].tolist() == [2, 0, 1]
    assert torch.all(batch["ids"][1] == -1)
    assert torch.count_nonzero(batch["masks"][1]) == 0
    assert torch.all(batch["ids"][2, 1] == -1)
    assert torch.count_nonzero(batch["masks"][2, 1]) == 0
    assert batch["masks"].dtype == torch.float32
    assert batch["mask_valid"].shape == (3, 2, 2)
    assert batch["mask_valid"][0].all()
    assert not batch["mask_valid"][1].any()
    assert not batch["mask_valid"][2, 1].any()


def test_collator_rejects_mismatched_frame_counts():
    with pytest.raises(ValueError, match="same frame count"):
        collate_entity_targets(
            [
                make_clip_target(1, frames=2),
                make_clip_target(1, frames=3),
            ]
        )


def test_collator_rejects_mismatched_padded_mask_sizes():
    with pytest.raises(ValueError, match="same padded mask size"):
        collate_entity_targets(
            [
                make_clip_target(1, size=8),
                make_clip_target(1, size=16),
            ]
        )


def test_collator_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty sequence"):
        collate_entity_targets([])



def test_collator_preserves_per_track_annotation_coverage():
    target = make_clip_target(1)
    target["ids"][0, 1] = -1
    target["masks"][0, 1].zero_()
    target["mask_valid"] = torch.tensor([[True, False]])

    batch = collate_entity_targets([target])

    assert batch["mask_valid"].tolist() == [[[True, False]]]
    assert batch["ids"].tolist() == [[[100, -1]]]
