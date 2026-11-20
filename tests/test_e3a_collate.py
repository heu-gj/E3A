import torch

from e3a.data.collate import collate_e3a_batch


def make_sample(track_count: int, *, frames: int = 2):
    ids = torch.full((track_count, frames), -1, dtype=torch.int64)
    masks = torch.zeros(track_count, frames, 8, 8)
    labels = torch.zeros(track_count, dtype=torch.int64)
    for index in range(track_count):
        ids[index] = 100 + index
        masks[index, :, 1:4, 1:4] = 1
    sample = {
        "images": torch.rand(frames, 3, 16, 16),
        "timestamps": torch.arange(frames).float() * 0.1,
        "entity_targets": {"labels": labels, "ids": ids, "masks": masks},
        "dataset": "hoi4d",
        "seq_name": f"sequence-{track_count}",
    }
    if track_count:
        sample["interaction_targets"] = {
            "contact": torch.zeros(frames, 2, track_count),
            "valid": torch.ones(frames, 2, track_count, dtype=torch.bool),
        }
    return sample


def test_collate_e3a_batch_pads_tracks_and_keeps_missing_labels_invalid():
    batch = collate_e3a_batch([make_sample(2), make_sample(0)])

    assert batch["images"].shape == (2, 2, 3, 16, 16)
    assert batch["timestamps"].shape == (2, 2)
    assert batch["entity_targets"]["ids"].shape == (2, 2, 2)
    assert batch["hand_targets"]["keypoints_camera"].shape == (2, 2, 2, 21, 3)
    assert batch["interaction_targets"]["contact"].shape == (2, 2, 2, 2)
    assert batch["anticipation_targets"]["action_valid"].tolist() == [False, False]
    assert not batch["hand_targets"]["valid"].any()
    assert not batch["interaction_targets"]["valid"][1].any()
    assert batch["dataset"] == ["hoi4d", "hoi4d"]


def test_collate_e3a_batch_preserves_explicit_anticipation_targets():
    sample = make_sample(1)
    sample["anticipation_targets"] = {
        "target_entity_track_id": 100,
        "target_entity_valid": True,
        "action_label": 3,
        "action_valid": True,
        "time_to_event": 0.75,
        "time_valid": True,
    }
    batch = collate_e3a_batch([sample])
    targets = batch["anticipation_targets"]
    assert targets["target_entity_track_ids"].tolist() == [100]
    assert targets["target_entity_valid"].tolist() == [True]
    assert targets["action_labels"].tolist() == [3]
    assert targets["action_valid"].tolist() == [True]
    torch.testing.assert_close(targets["time_to_event"], torch.tensor([0.75]))
