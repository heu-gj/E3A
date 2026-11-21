import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from e3a.data.hoi4d import hoi4d_color_palette
from e3a.data.hoi4d_dataset import (
    HOI4DDataset,
    _NumpyArraysOnlyUnpickler,
    _decode_mask_with_validity,
    collate_hoi4d_batch,
    make_hoi4d_dataloader,
    resolve_hoi4d_sequence_paths,
)


SEQUENCE = "ZY20210800001/H1/C20/N30/S295/s04/T4"


class FakeVideoCapture:
    registry = {}
    instances = []

    def __init__(self, path):
        self.path = str(path)
        self.frames, self.metadata = self.registry[self.path]
        self.position = 0
        self.convert_rgb = 1
        self.released = False
        self.instances.append(self)

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.metadata["width"]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.metadata["height"]
        if prop == cv2.CAP_PROP_FPS:
            return self.metadata["fps"]
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        elif prop == cv2.CAP_PROP_CONVERT_RGB:
            self.convert_rgb = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


def _make_synthetic_root(tmp_path: Path, monkeypatch, *, omit_masks=False):
    FakeVideoCapture.registry = {}
    FakeVideoCapture.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)

    paths = resolve_hoi4d_sequence_paths(tmp_path, SEQUENCE)
    paths.rgb_video.parent.mkdir(parents=True, exist_ok=True)
    paths.depth_video.parent.mkdir(parents=True, exist_ok=True)
    paths.rgb_video.touch()
    paths.depth_video.touch()
    paths.mask_dir.mkdir(parents=True, exist_ok=True)
    paths.objpose_dir.mkdir(parents=True, exist_ok=True)
    paths.right_hand_dir.mkdir(parents=True, exist_ok=True)
    paths.left_hand_dir.mkdir(parents=True, exist_ok=True)
    paths.intrinsics_file.parent.mkdir(parents=True, exist_ok=True)

    height, width, frame_count = 24, 32, 4
    rgb_frames = []
    depth_frames = []
    for index in range(frame_count):
        bgr = np.zeros((height, width, 3), dtype=np.uint8)
        bgr[..., 2] = 255
        bgr[:, :, 1] = index
        rgb_frames.append(bgr)
        depth_frames.append(
            np.full((height, width), 1000 + index, dtype=np.uint16)
        )
    metadata = {"width": width, "height": height, "fps": 15.0}
    FakeVideoCapture.registry[str(paths.rgb_video)] = (rgb_frames, metadata)
    FakeVideoCapture.registry[str(paths.depth_video)] = (depth_frames, metadata)

    palette = hoi4d_color_palette()
    if not omit_masks:
        for index in range(frame_count):
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask[4:12, 5:15] = palette[1]
            mask[12:16, 5:10] = palette[2]
            if index == 2:
                mask[0, 0] = palette[4]
            Image.fromarray(mask).save(paths.mask_dir / f"{index:05d}.png")

    for index in range(frame_count):
        record = {
            "frameId": 1 if index in (0, 1) else index,
            "isEffective": 1,
            "dataList": [
                {
                    "id": 1,
                    "label": "chair",
                    "center": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "dimensions": {"length": 1.0, "width": 1.0, "height": 1.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
            ],
        }
        (paths.objpose_dir / f"{index}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    action = {
        "info": {"duration": frame_count / 15.0},
        "events": [
            {"event": "grasp", "startTime": 0.1, "endTime": 0.12},
            {"event": "release", "startTime": 0.2, "endTime": 0.22},
        ],
    }
    paths.action_file.parent.mkdir(parents=True, exist_ok=True)
    paths.action_file.write_text(json.dumps(action), encoding="utf-8")
    np.save(
        paths.intrinsics_file,
        np.asarray([[100.0, 0.0, 16.0], [0.0, 100.0, 12.0], [0.0, 0.0, 1.0]]),
    )

    mano_record = {
        "poseCoeff": np.zeros(48, dtype=np.float32),
        "beta": np.zeros(10, dtype=np.float32),
        "trans": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "kps2D": np.ones((21, 2), dtype=np.float32) * 4,
    }
    with (paths.right_hand_dir / "0.pickle").open("wb") as stream:
        pickle.dump(mano_record, stream)
    with (paths.left_hand_dir / "1.pickle").open("wb") as stream:
        pickle.dump(mano_record, stream)
    return paths


def test_resolve_paths_rejects_traversal_and_uses_audited_layout(tmp_path):
    paths = resolve_hoi4d_sequence_paths(tmp_path, SEQUENCE)
    assert paths.rgb_video == (
        tmp_path / "extracted/release/HOI4D_release" / SEQUENCE / "align_rgb/image.mp4"
    )
    assert paths.depth_video == (
        tmp_path / "extracted/depth/HOI4D_depth_video" / SEQUENCE / "align_depth/depth_video.avi"
    )
    with pytest.raises(ValueError, match="traversal"):
        resolve_hoi4d_sequence_paths(tmp_path, "../escape/H1/C20/N30/S295/s04/T4")
    with pytest.raises(ValueError, match="seven components"):
        resolve_hoi4d_sequence_paths(tmp_path, "C20/N30/T4")


def test_unknown_palette_pixels_are_invalid_but_known_mask_pixels_survive():
    palette = hoi4d_color_palette()
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[0, 0] = palette[1]
    rgb[0, 1] = palette[2]
    rgb[0, 2] = palette[4]

    semantic, instance, valid = _decode_mask_with_validity(rgb, "C20")
    assert semantic[0, 0] == 2
    assert instance[0, 0] == 1
    assert semantic[0, 1] == 47
    assert valid[0, 0]
    assert valid[0, 1]
    assert not valid[0, 2]
    assert semantic[0, 2] == -1
    assert instance[0, 2] == -1


def test_dataset_reads_rgb_depth_masks_pose_action_and_raw_mano(tmp_path, monkeypatch):
    _make_synthetic_root(tmp_path, monkeypatch)
    dataset = HOI4DDataset(
        tmp_path,
        SEQUENCE,
        clip_length=2,
        clip_stride=2,
        image_size=(12, 16),
        action_vocabulary={"hoi4d::C20::grasp": 3},
    )

    sample = dataset[0]
    assert sample["images"].shape == (2, 3, 12, 16)
    assert sample["images"][0, 0].mean().item() == pytest.approx(1.0)
    assert sample["depth"].shape == (2, 1, 12, 16)
    assert sample["depth"][0].mean().item() == pytest.approx(1000.0)
    assert sample["depth_units"] == "unknown"
    assert sample["timestamps"].tolist() == pytest.approx([0.0, 1.0 / 15.0])
    assert sample["frame_indices"].tolist() == [0, 1]

    assert sample["intrinsics_valid"]
    assert sample["intrinsics"][0, 0].item() == pytest.approx(50.0)
    assert sample["intrinsics"][0, 2].item() == pytest.approx(8.0)
    assert sample["intrinsics"][1, 1].item() == pytest.approx(50.0)
    assert sample["intrinsics"][1, 2].item() == pytest.approx(6.0)

    assert sample["entity_targets"]["ids"].shape == (1, 2)
    assert sample["entity_targets"]["mask_valid"].all()
    assert sample["entity_targets"]["pixel_valid"].all()
    assert sample["entity_supervision_valid"]

    assert sample["objpose_frame_ids"].tolist() == [1, 1]
    assert sample["objpose_records"][0]["frameId"] == 1
    assert sample["metadata"]["objpose_frame_id_status"].startswith("raw source")

    mano = sample["mano_parameters"]
    assert mano["valid"].tolist() == [[True, False], [False, True]]
    assert mano["pose_coefficients"].shape == (2, 2, 48)
    assert mano["keypoints_2d"][0, 0, 0].tolist() == pytest.approx([2.0, 2.0])
    assert "no MANO forward" in sample["metadata"]["mano_status"]

    anticipation = sample["anticipation_targets"]
    assert anticipation["target_entity_valid"] is False
    assert anticipation["action_label"] == 3
    assert anticipation["action_valid"] is True
    assert anticipation["time_valid"] is True


def test_dataset_marks_unknown_pixels_and_missing_entity_supervision(tmp_path, monkeypatch):
    _make_synthetic_root(tmp_path, monkeypatch)
    dataset = HOI4DDataset(
        tmp_path,
        SEQUENCE,
        clip_length=2,
        clip_stride=2,
        load_mano=False,
    )
    sample = dataset[1]
    assert sample["frame_indices"].tolist() == [2, 3]
    assert not sample["entity_targets"]["pixel_valid"][0, 0, 0].item()
    assert sample["entity_targets"]["pixel_valid"][1].all()
    assert sample["entity_supervision_valid"]
    assert sample["entity_targets"]["mask_valid"].all()


def test_empty_mask_clip_is_not_marked_as_negative_entity_supervision(tmp_path, monkeypatch):
    _make_synthetic_root(tmp_path, monkeypatch, omit_masks=True)
    dataset = HOI4DDataset(
        tmp_path,
        SEQUENCE,
        clip_length=2,
        load_mano=False,
    )
    sample = dataset[0]
    assert sample["entity_targets"]["labels"].numel() == 0
    assert sample["entity_targets"]["masks"].shape == (0, 2, 24, 32)
    assert not sample["entity_targets"]["pixel_valid"].any()
    assert not sample["entity_supervision_valid"]


def test_collator_preserves_pixel_depth_intrinsics_and_clip_validity(tmp_path, monkeypatch):
    _make_synthetic_root(tmp_path, monkeypatch)
    dataset = HOI4DDataset(
        tmp_path,
        SEQUENCE,
        clip_length=2,
        clip_stride=2,
        load_mano=True,
    )
    first, second = dataset[0], dataset[1]
    batch = collate_hoi4d_batch([first, second])
    assert batch["images"].shape == (2, 2, 3, 24, 32)
    assert batch["depth"].shape == (2, 2, 1, 24, 32)
    assert batch["intrinsics"].shape == (2, 3, 3)
    assert batch["frame_indices"].tolist() == [[0, 1], [2, 3]]
    assert batch["entity_targets"]["pixel_valid"].shape == (2, 2, 24, 32)
    assert not batch["entity_targets"]["pixel_valid"][1, 0, 0, 0].item()
    assert batch["entity_supervision_valid"].tolist() == [True, True]
    assert batch["mano_parameters"]["valid"].shape == (2, 2, 2)


def test_make_dataloader_uses_hoi4d_collator(tmp_path, monkeypatch):
    _make_synthetic_root(tmp_path, monkeypatch)
    dataset = HOI4DDataset(tmp_path, SEQUENCE, clip_length=2, clip_stride=2)
    loader = make_hoi4d_dataloader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch["images"].shape == (2, 2, 3, 24, 32)
    assert batch["dataset"] == ["hoi4d", "hoi4d"]


def test_restricted_mano_unpickler_rejects_non_numpy_globals():
    payload = pickle.dumps(Path)
    with pytest.raises(pickle.UnpicklingError, match="Disallowed global"):
        _NumpyArraysOnlyUnpickler(
            __import__("io").BytesIO(payload), encoding="latin1"
        ).load()
