"""Raw-video HOI4D Dataset and DataLoader integration.

The reader preserves source indices and unknown supervision. It does not infer
physical units, object-pose coordinate frames, persistent source identities, or
MANO 3D joints from the raw parameters.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from e3a.data.collate import collate_e3a_batch
from e3a.data.hoi4d_supervision import HOI4DSupervisionBuilder
from e3a.data.hoi4d import (
    HOI4D_2D_CATEGORIES,
    build_hoi4d_anticipation_target,
    build_hoi4d_entity_clip_targets,
    decode_hoi4d_motion_mask,
    hoi4d_color_palette,
    load_hoi4d_action_segments,
)


@dataclass(frozen=True)
class HOI4DSequencePaths:
    """Resolved paths for one source sequence."""

    sequence_id: str
    rgb_video: Path
    depth_video: Path
    annotation_root: Path
    mask_dir: Path
    objpose_dir: Path
    action_file: Path
    right_hand_dir: Path
    left_hand_dir: Path
    intrinsics_file: Path


def _normalize_sequence_id(sequence_id: str) -> str:
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("sequence_id must be a non-empty relative path.")
    if "\\" in sequence_id:
        raise ValueError("sequence_id must use POSIX '/' separators.")
    path = PurePosixPath(sequence_id)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("sequence_id must not be absolute or contain traversal segments.")
    if len(path.parts) != 7:
        raise ValueError(
            "HOI4D sequence_id must have seven components, e.g. "
            "ZY20210800001/H1/C20/N30/S295/s04/T4."
        )
    return path.as_posix()


def resolve_hoi4d_sequence_paths(
    root: str | Path, sequence_id: str
) -> HOI4DSequencePaths:
    """Resolve the audited HOI4D release layout without guessing alternate paths."""
    root_path = Path(root).expanduser().resolve()
    normalized = _normalize_sequence_id(sequence_id)
    parts = PurePosixPath(normalized).parts
    sequence = Path(*parts)
    camera_id = parts[0]
    annotation_root = (
        root_path / "extracted/annotations/HOI4D_annotations" / sequence
    )
    hand_root = root_path / "extracted/hand_pose/Hand_pose"
    return HOI4DSequencePaths(
        sequence_id=normalized,
        rgb_video=(
            root_path
            / "extracted/release/HOI4D_release"
            / sequence
            / "align_rgb/image.mp4"
        ),
        depth_video=(
            root_path
            / "extracted/depth/HOI4D_depth_video"
            / sequence
            / "align_depth/depth_video.avi"
        ),
        annotation_root=annotation_root,
        mask_dir=annotation_root / "2Dseg/mask",
        objpose_dir=annotation_root / "objpose",
        action_file=annotation_root / "action/color.json",
        right_hand_dir=hand_root / "handpose_right_hand" / sequence,
        left_hand_dir=hand_root / "handpose_left_hand" / sequence,
        intrinsics_file=(
            root_path
            / "extracted/camera_params/camera_params"
            / camera_id
            / "intrin.npy"
        ),
    )


def _probe_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open HOI4D video: {path}")
        metadata = {
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        }
    finally:
        capture.release()
    if (
        metadata["width"] <= 0
        or metadata["height"] <= 0
        or metadata["frame_count"] <= 0
        or not np.isfinite(metadata["fps"])
        or metadata["fps"] <= 0
    ):
        raise RuntimeError(f"Invalid HOI4D video metadata for {path}: {metadata}")
    return metadata


def _read_video_frames(
    path: Path, frame_indices: Sequence[int], *, depth: bool = False
) -> list[np.ndarray]:
    """Read requested monotonically increasing frame indices from one video.

    Seeking is attempted once per clip. If the backend rejects or misreports
    the seek position, decoding falls back to frame zero. Depth decoding keeps
    OpenCV's RGB conversion disabled, matching the audited HOI4D reader.
    """
    indices = [int(index) for index in frame_indices]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("frame_indices must be a non-empty list of non-negative integers.")
    if indices != sorted(set(indices)):
        raise ValueError("frame_indices must be strictly increasing and unique.")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open HOI4D video: {path}")
        if depth:
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        requested_start = indices[0]
        decode_start = requested_start
        if requested_start:
            seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, requested_start)
            reported_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not seek_ok or abs(reported_position - requested_start) > 0.5:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                decode_start = 0

        wanted = set(indices)
        selected: dict[int, np.ndarray] = {}
        for frame_index in range(decode_start, indices[-1] + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"HOI4D video ended before requested frame {frame_index}: {path}"
                )
            if frame_index not in wanted:
                continue
            value = np.asarray(frame)
            if depth:
                if value.ndim == 3 and value.shape[-1] == 1:
                    value = value[..., 0]
                if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
                    raise TypeError(
                        "HOI4D depth must decode as a single-channel integer array "
                        f"when RGB conversion is disabled; got {value.shape}/{value.dtype}."
                    )
            else:
                if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                    raise TypeError(
                        f"HOI4D RGB must decode as uint8 BGR [H,W,3], got {value.shape}/{value.dtype}."
                    )
                value = np.ascontiguousarray(value[..., ::-1])
            selected[frame_index] = value.copy()

        if len(selected) != len(indices):
            raise RuntimeError(
                f"Decoded {len(selected)} of {len(indices)} requested frames from {path}."
            )
        return [selected[index] for index in indices]
    finally:
        capture.release()


def _resize_array(
    array: np.ndarray, image_size: tuple[int, int], *, interpolation: int
) -> np.ndarray:
    height, width = image_size
    return cv2.resize(array, (width, height), interpolation=interpolation)


def _resize_pixel_validity(
    pixel_valid: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Conservatively resize pixel validity when downsampling image-space labels."""
    source_height, source_width = pixel_valid.shape
    target_height, target_width = image_size
    if (source_height, source_width) == image_size:
        return pixel_valid.copy()
    if target_height <= source_height and target_width <= source_width:
        invalid_fraction = cv2.resize(
            (~pixel_valid).astype(np.float32),
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return invalid_fraction == 0.0
    return cv2.resize(
        pixel_valid.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.bool_)


def _decode_mask_with_validity(
    rgb_mask: np.ndarray, category: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode known official palette colors and mark every unknown pixel invalid.

    The existing public decoder remains strict and continues to raise on an
    unknown color. This reader decodes unique colors independently so known
    labels survive a partially unknown frame such as audited C20 frame 220.
    """
    if category not in HOI4D_2D_CATEGORIES:
        raise ValueError(f"Unsupported HOI4D 2D motion-segmentation category: {category!r}")
    image = np.asarray(rgb_mask)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise TypeError("rgb_mask must be a uint8 RGB array with shape [H,W,3].")

    colors, inverse = np.unique(
        image.reshape(-1, 3), axis=0, return_inverse=True
    )
    semantic_by_color = np.full(len(colors), -1, dtype=np.int16)
    instance_by_color = np.full(len(colors), -1, dtype=np.int16)
    valid_by_color = np.zeros(len(colors), dtype=np.bool_)

    for color_index, color in enumerate(colors):
        try:
            semantic, instance = decode_hoi4d_motion_mask(
                color.reshape(1, 1, 3), category
            )
        except ValueError:
            continue
        semantic_by_color[color_index] = semantic[0, 0]
        instance_by_color[color_index] = instance[0, 0]
        valid_by_color[color_index] = True

    shape = image.shape[:2]
    return (
        semantic_by_color[inverse].reshape(shape),
        instance_by_color[inverse].reshape(shape),
        valid_by_color[inverse].reshape(shape),
    )


class _NumpyArraysOnlyUnpickler(pickle.Unpickler):
    """Restrict HOI4D MANO pickle globals to NumPy array reconstruction."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(
                f"Disallowed global in HOI4D MANO pickle: {module}.{name}"
            )
        if module.endswith("multiarray") and name == "_reconstruct":
            return np.core.multiarray._reconstruct
        if module == "numpy" and name == "ndarray":
            return np.ndarray
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise pickle.UnpicklingError(f"Disallowed global in HOI4D MANO pickle: {module}.{name}")


def _read_mano_record(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as stream:
        record = _NumpyArraysOnlyUnpickler(stream, encoding="latin1").load()
    if not isinstance(record, Mapping):
        raise TypeError(f"HOI4D MANO record must be a mapping: {path}")

    expected_shapes = {
        "poseCoeff": (48,),
        "beta": (10,),
        "trans": (3,),
        "kps2D": (21, 2),
    }
    output: dict[str, np.ndarray] = {}
    for key, shape in expected_shapes.items():
        if key not in record:
            raise ValueError(f"HOI4D MANO record is missing {key!r}: {path}")
        value = np.asarray(record[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(
                f"HOI4D MANO {key} must be finite with shape {shape}; "
                f"got {value.shape}: {path}"
            )
        output[key] = value
    return output


def _read_json_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise TypeError(f"HOI4D JSON root must be an object: {path}")
    return dict(value)


def _empty_entity_targets(
    frame_count: int, height: int, width: int
) -> dict[str, torch.Tensor]:
    return {
        "labels": torch.zeros((0,), dtype=torch.int64),
        "ids": torch.full((0, frame_count), -1, dtype=torch.int64),
        "masks": torch.zeros((0, frame_count, height, width), dtype=torch.float32),
        "mask_valid": torch.zeros((0, frame_count), dtype=torch.bool),
    }


class HOI4DDataset(Dataset[dict[str, Any]]):
    """Read source HOI4D clips as E3A-compatible samples.

    Args:
        root: Local HOI4D root containing the extracted directory layout.
        sequence_ids: Explicit sequence IDs. Split by sequence/subject before
            constructing datasets; never randomly split neighboring clips.
        clip_length: Number of sampled frames in each clip.
        frame_stride: Source-frame interval between clip frames.
        clip_stride: Source-frame interval between clip start positions.
        image_size: Optional exact (height, width) resize for all image-space
            modalities. RGB uses area interpolation; depth/masks use nearest.
            Intrinsics and MANO 2D keypoints are scaled accordingly.
        load_depth: Decode the aligned depth video with RGB conversion disabled.
        load_objpose: Preserve raw per-frame objpose JSON records.
        load_mano: Preserve raw MANO parameters and source 2D keypoints. This
            does not run MANO forward kinematics or create 3D hand joints.
        action_vocabulary: Training-split-only dataset-prefixed vocabulary.
        strict_video_metadata: Require RGB/depth frame count, FPS, and image
            dimensions to agree. This is a metadata check, not pixel calibration.
    """

    def __init__(
        self,
        root: str | Path,
        sequence_ids: str | Sequence[str],
        *,
        clip_length: int = 8,
        frame_stride: int = 1,
        clip_stride: int = 1,
        image_size: tuple[int, int] | None = None,
        load_depth: bool = True,
        load_objpose: bool = True,
        load_mano: bool = True,
        action_vocabulary: Mapping[str, int] | None = None,
        strict_video_metadata: bool = True,
    ) -> None:
        if isinstance(sequence_ids, str):
            sequence_ids = [sequence_ids]
        if not isinstance(sequence_ids, Sequence) or not sequence_ids:
            raise ValueError("sequence_ids must be a non-empty sequence.")
        if clip_length <= 0 or frame_stride <= 0 or clip_stride <= 0:
            raise ValueError("clip_length, frame_stride, and clip_stride must be positive.")
        if image_size is not None:
            if (
                len(image_size) != 2
                or int(image_size[0]) <= 0
                or int(image_size[1]) <= 0
            ):
                raise ValueError("image_size must be a positive (height, width) pair.")
            image_size = (int(image_size[0]), int(image_size[1]))
        if action_vocabulary is not None:
            for key, value in action_vocabulary.items():
                if (
                    not isinstance(key, str)
                    or isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                ):
                    raise ValueError(
                        "action_vocabulary must map strings to non-negative integers."
                    )

        self.root = Path(root).expanduser().resolve()
        self.clip_length = int(clip_length)
        self.frame_stride = int(frame_stride)
        self.clip_stride = int(clip_stride)
        self.image_size = image_size
        self.load_depth = bool(load_depth)
        self.load_objpose = bool(load_objpose)
        self.load_mano = bool(load_mano)
        self.action_vocabulary = dict(action_vocabulary or {})
        self.sequence_ids = tuple(_normalize_sequence_id(value) for value in sequence_ids)
        if len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise ValueError("sequence_ids must not contain duplicates.")

        self._sequences: list[dict[str, Any]] = []
        self._samples: list[tuple[int, int]] = []
        clip_span = (self.clip_length - 1) * self.frame_stride + 1

        for sequence_index, sequence_id in enumerate(self.sequence_ids):
            paths = resolve_hoi4d_sequence_paths(self.root, sequence_id)
            if not paths.rgb_video.is_file():
                raise FileNotFoundError(f"Missing HOI4D RGB video: {paths.rgb_video}")
            if not paths.mask_dir.is_dir():
                raise FileNotFoundError(f"Missing HOI4D motion-mask directory: {paths.mask_dir}")

            rgb_meta = _probe_video(paths.rgb_video)
            if self.load_depth:
                if not paths.depth_video.is_file():
                    raise FileNotFoundError(f"Missing HOI4D depth video: {paths.depth_video}")
                depth_meta = _probe_video(paths.depth_video)
                if strict_video_metadata:
                    for field in ("width", "height", "frame_count"):
                        if depth_meta[field] != rgb_meta[field]:
                            raise ValueError(
                                f"HOI4D RGB/depth {field} mismatch for {sequence_id}: "
                                f"{rgb_meta[field]} vs {depth_meta[field]}"
                            )
                    if abs(float(depth_meta["fps"]) - float(rgb_meta["fps"])) > 1e-3:
                        raise ValueError(
                            f"HOI4D RGB/depth FPS mismatch for {sequence_id}: "
                            f"{rgb_meta['fps']} vs {depth_meta['fps']}"
                        )

            category = PurePosixPath(sequence_id).parts[2]
            if category not in HOI4D_2D_CATEGORIES:
                raise ValueError(
                    f"Sequence {sequence_id} uses unsupported HOI4D 2D category {category!r}."
                )
            segments = (
                load_hoi4d_action_segments(paths.action_file, category=category)
                if paths.action_file.is_file()
                else []
            )
            intrinsics = None
            if paths.intrinsics_file.is_file():
                intrinsics = np.asarray(
                    np.load(paths.intrinsics_file, allow_pickle=False), dtype=np.float32
                )
                if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                    raise ValueError(
                        f"HOI4D intrinsics must be finite [3,3]: {paths.intrinsics_file}"
                    )

            frame_count = int(rgb_meta["frame_count"])
            if clip_span > frame_count:
                raise ValueError(
                    f"clip span {clip_span} exceeds {sequence_id} frame count {frame_count}."
                )
            self._sequences.append(
                {
                    "paths": paths,
                    "rgb_meta": rgb_meta,
                    "category": category,
                    "segments": segments,
                    "intrinsics": intrinsics,
                }
            )
            last_start = frame_count - clip_span
            starts = list(range(0, last_start + 1, self.clip_stride))
            if starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                self._samples.append((sequence_index, start))

        if not self._samples:
            raise ValueError("No valid HOI4D clips were generated from the supplied sequences.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, start = self._samples[index]
        sequence = self._sequences[sequence_index]
        paths: HOI4DSequencePaths = sequence["paths"]
        rgb_meta = sequence["rgb_meta"]
        frame_indices = [
            start + offset * self.frame_stride for offset in range(self.clip_length)
        ]
        rgb_frames = _read_video_frames(paths.rgb_video, frame_indices, depth=False)
        source_height = int(rgb_meta["height"])
        source_width = int(rgb_meta["width"])
        target_size = self.image_size or (source_height, source_width)

        rgb_tensor_frames = []
        for frame in rgb_frames:
            if frame.shape[:2] != (source_height, source_width):
                raise ValueError(
                    f"Decoded RGB size {frame.shape[:2]} does not match video metadata "
                    f"{(source_height, source_width)} for {paths.sequence_id}."
                )
            if target_size != (source_height, source_width):
                frame = _resize_array(
                    frame, target_size, interpolation=cv2.INTER_AREA
                )
            rgb_tensor_frames.append(frame)
        images_array = np.stack(rgb_tensor_frames).astype(np.float32) / 255.0
        images = torch.from_numpy(images_array).permute(0, 3, 1, 2).contiguous()

        depth = None
        depth_valid = None
        if self.load_depth:
            depth_frames = _read_video_frames(
                paths.depth_video, frame_indices, depth=True
            )
            resized_depth = []
            for frame in depth_frames:
                if frame.shape[:2] != (source_height, source_width):
                    raise ValueError(
                        f"Decoded depth size {frame.shape[:2]} does not match RGB "
                        f"{(source_height, source_width)} for {paths.sequence_id}."
                    )
                if target_size != (source_height, source_width):
                    frame = _resize_array(
                        frame, target_size, interpolation=cv2.INTER_NEAREST
                    )
                resized_depth.append(frame)
            depth_array = np.stack(resized_depth).astype(np.float32)
            depth_valid_array = np.isfinite(depth_array) & (depth_array > 0)
            depth = torch.from_numpy(depth_array[:, None].copy())
            depth_valid = torch.from_numpy(depth_valid_array[:, None].copy())

        mask_rgb_frames: list[np.ndarray | None] = []
        pixel_valid_frames: list[np.ndarray] = []
        palette_background = hoi4d_color_palette(1)[0]
        for frame_index in frame_indices:
            mask_path = paths.mask_dir / f"{frame_index:05d}.png"
            if not mask_path.is_file():
                mask_rgb_frames.append(None)
                pixel_valid_frames.append(np.zeros(target_size, dtype=np.bool_))
                continue
            with Image.open(mask_path) as source:
                mask_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            if mask_rgb.shape[:2] != (source_height, source_width):
                raise ValueError(
                    f"Mask {mask_path} has size {mask_rgb.shape[:2]}, expected "
                    f"{(source_height, source_width)}."
                )
            _, _, source_pixel_valid = _decode_mask_with_validity(
                mask_rgb, sequence["category"]
            )
            if target_size != (source_height, source_width):
                mask_rgb = _resize_array(
                    mask_rgb, target_size, interpolation=cv2.INTER_NEAREST
                )
                pixel_valid = _resize_pixel_validity(source_pixel_valid, target_size)
            else:
                pixel_valid = source_pixel_valid
            safe_mask = mask_rgb.copy()
            safe_mask[~pixel_valid] = palette_background
            mask_rgb_frames.append(safe_mask)
            pixel_valid_frames.append(pixel_valid)

        if any(frame is not None for frame in mask_rgb_frames):
            entity_targets = build_hoi4d_entity_clip_targets(
                mask_rgb_frames, sequence["category"]
            )
        else:
            entity_targets = _empty_entity_targets(
                self.clip_length, target_size[0], target_size[1]
            )
        entity_targets["pixel_valid"] = torch.from_numpy(
            np.stack(pixel_valid_frames).astype(np.bool_)
        )
        # The existing clip flag below only states that some positive masks
        # are usable. HOI4D masks do not certify complete scene enumeration or
        # explicit empty-scene annotation. Preserve that flag's old meaning and
        # separately withhold No Object coverage, without dropping positives.
        entity_targets["entity_classification_valid"] = torch.zeros(
            self.clip_length, dtype=torch.bool
        )
        entity_supervision_valid = bool(
            entity_targets["mask_valid"].any().item()
            and entity_targets["labels"].numel() > 0
        )

        intrinsics = torch.zeros((3, 3), dtype=torch.float32)
        intrinsics_valid = sequence["intrinsics"] is not None
        if intrinsics_valid:
            intrinsics_array = sequence["intrinsics"].copy()
            scale_x = target_size[1] / source_width
            scale_y = target_size[0] / source_height
            intrinsics_array[0, :] *= scale_x
            intrinsics_array[1, :] *= scale_y
            intrinsics = torch.from_numpy(intrinsics_array.astype(np.float32))

        objpose_records: list[dict[str, Any] | None] = []
        objpose_frame_ids = []
        objpose_record_present = []
        for frame_index in frame_indices:
            objpose_path = paths.objpose_dir / f"{frame_index}.json"
            if self.load_objpose and objpose_path.is_file():
                record = _read_json_mapping(objpose_path)
                objpose_records.append(record)
                raw_frame_id = record.get("frameId")
                if isinstance(raw_frame_id, (int, float)) and float(raw_frame_id).is_integer():
                    objpose_frame_ids.append(int(raw_frame_id))
                else:
                    objpose_frame_ids.append(-1)
                objpose_record_present.append(True)
            else:
                objpose_records.append(None)
                objpose_frame_ids.append(-1)
                objpose_record_present.append(False)

        mano_parameters = {
            "pose_coefficients": torch.zeros(
                self.clip_length, 2, 48, dtype=torch.float32
            ),
            "betas": torch.zeros(self.clip_length, 2, 10, dtype=torch.float32),
            "translations": torch.zeros(self.clip_length, 2, 3, dtype=torch.float32),
            "keypoints_2d": torch.zeros(
                self.clip_length, 2, 21, 2, dtype=torch.float32
            ),
            "valid": torch.zeros(self.clip_length, 2, dtype=torch.bool),
            "side_order": ("right", "left"),
        }
        if self.load_mano:
            hand_directories = (paths.right_hand_dir, paths.left_hand_dir)
            for frame_position, frame_index in enumerate(frame_indices):
                for hand_index, hand_directory in enumerate(hand_directories):
                    mano_path = hand_directory / f"{frame_index}.pickle"
                    if not mano_path.is_file():
                        continue
                    record = _read_mano_record(mano_path)
                    mano_parameters["pose_coefficients"][frame_position, hand_index] = (
                        torch.from_numpy(record["poseCoeff"])
                    )
                    mano_parameters["betas"][frame_position, hand_index] = (
                        torch.from_numpy(record["beta"])
                    )
                    mano_parameters["translations"][frame_position, hand_index] = (
                        torch.from_numpy(record["trans"])
                    )
                    keypoints = record["kps2D"].copy()
                    keypoints[:, 0] *= target_size[1] / source_width
                    keypoints[:, 1] *= target_size[0] / source_height
                    mano_parameters["keypoints_2d"][frame_position, hand_index] = (
                        torch.from_numpy(keypoints)
                    )
                    mano_parameters["valid"][frame_position, hand_index] = True

        fps = float(rgb_meta["fps"])
        timestamps = torch.tensor(
            [frame_index / fps for frame_index in frame_indices], dtype=torch.float32
        )
        anticipation_targets = build_hoi4d_anticipation_target(
            sequence["segments"],
            observation_end_time=float(timestamps[-1]),
            action_vocabulary=self.action_vocabulary,
        )

        return {
            "images": images,
            "timestamps": timestamps,
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "depth": depth,
            "depth_valid": depth_valid,
            "depth_units": "unknown",
            "intrinsics": intrinsics,
            "intrinsics_valid": intrinsics_valid,
            "entity_targets": entity_targets,
            "entity_supervision_valid": entity_supervision_valid,
            "mano_parameters": mano_parameters,
            "objpose_records": objpose_records,
            "objpose_frame_ids": torch.tensor(objpose_frame_ids, dtype=torch.int64),
            "objpose_record_present": torch.tensor(
                objpose_record_present, dtype=torch.bool
            ),
            "action_segments": [dict(segment) for segment in sequence["segments"]],
            "anticipation_targets": anticipation_targets,
            "dataset": "hoi4d",
            "seq_name": paths.sequence_id,
            "metadata": {
                "sequence_id": paths.sequence_id,
                "category": sequence["category"],
                "camera_id": PurePosixPath(paths.sequence_id).parts[0],
                "fps": fps,
                "video_frame_count": int(rgb_meta["frame_count"]),
                "timestamp_basis": "decoded_frame_index / RGB video FPS; no sensor timestamp",
                "entity_track_id_status": "provisional category-local 2D motion-mask instance ID",
                "objpose_frame_id_status": "raw source field preserved; not used as video index",
                "objpose_coordinate_frame": "unknown",
                "objpose_units": "unknown",
                "depth_units": "unknown",
                "mano_status": "raw parameters and 2D keypoints only; no MANO forward reconstruction",
            },
        }


def collate_hoi4d_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate HOI4D samples and preserve modality/annotation validity metadata."""
    batch = collate_e3a_batch(samples)
    _, frame_count, _, height, width = batch["images"].shape
    device = batch["images"].device

    # Generic collation now preserves pixel validity; do not copy/stack it twice.
    if batch["entity_targets"]["pixel_valid"].shape != (
        len(samples), frame_count, height, width
    ):
        raise ValueError("HOI4D pixel_valid must align with image [B,T,H,W].")

    if any(sample.get("depth") is not None for sample in samples):
        depth_values = []
        depth_valid_values = []
        for sample_index, sample in enumerate(samples):
            value = sample.get("depth")
            valid = sample.get("depth_valid")
            if value is None:
                value = torch.zeros(frame_count, 1, height, width, dtype=torch.float32)
                valid = torch.zeros_like(value, dtype=torch.bool)
            if not torch.is_tensor(value) or tuple(value.shape) != (
                frame_count, 1, height, width
            ):
                raise ValueError(f"samples[{sample_index}] depth must have shape [T,1,H,W].")
            if valid is None:
                valid = torch.isfinite(value) & (value > 0)
            if (
                not torch.is_tensor(valid)
                or valid.dtype != torch.bool
                or tuple(valid.shape) != tuple(value.shape)
            ):
                raise ValueError(
                    f"samples[{sample_index}] depth_valid must be bool [T,1,H,W]."
                )
            depth_values.append(value.to(device=device, dtype=torch.float32))
            depth_valid_values.append(valid.to(device=device))
        batch["depth"] = torch.stack(depth_values, dim=0)
        batch["depth_valid"] = torch.stack(depth_valid_values, dim=0)
        batch["depth_units"] = [str(sample.get("depth_units", "unknown")) for sample in samples]

    if any(sample.get("intrinsics") is not None for sample in samples):
        intrinsics_values = []
        intrinsics_valid = []
        for sample_index, sample in enumerate(samples):
            value = sample.get("intrinsics")
            valid = bool(sample.get("intrinsics_valid", value is not None))
            if value is None:
                value = torch.zeros(3, 3, dtype=torch.float32)
                valid = False
            if not torch.is_tensor(value) or tuple(value.shape) != (3, 3):
                raise ValueError(f"samples[{sample_index}] intrinsics must have shape [3,3].")
            if valid and not torch.isfinite(value).all():
                raise ValueError(f"samples[{sample_index}] intrinsics must be finite.")
            intrinsics_values.append(value.to(device=device, dtype=torch.float32))
            intrinsics_valid.append(valid)
        batch["intrinsics"] = torch.stack(intrinsics_values, dim=0)
        batch["intrinsics_valid"] = torch.tensor(
            intrinsics_valid, dtype=torch.bool, device=device
        )

    if any(sample.get("frame_indices") is not None for sample in samples):
        indices = []
        for sample_index, sample in enumerate(samples):
            value = sample.get("frame_indices")
            if value is None:
                value = torch.full((frame_count,), -1, dtype=torch.int64)
            if not torch.is_tensor(value) or tuple(value.shape) != (frame_count,):
                raise ValueError(
                    f"samples[{sample_index}] frame_indices must have shape [T]."
                )
            indices.append(value.to(device=device, dtype=torch.int64))
        batch["frame_indices"] = torch.stack(indices, dim=0)

    if any(sample.get("mano_parameters") is not None for sample in samples):
        fields = {
            "pose_coefficients": (frame_count, 2, 48),
            "betas": (frame_count, 2, 10),
            "translations": (frame_count, 2, 3),
            "keypoints_2d": (frame_count, 2, 21, 2),
            "valid": (frame_count, 2),
        }
        mano_batch: dict[str, Any] = {}
        for field, shape in fields.items():
            dtype = torch.bool if field == "valid" else torch.float32
            values = []
            for sample_index, sample in enumerate(samples):
                parameters = sample.get("mano_parameters")
                value = None if parameters is None else parameters.get(field)
                if value is None:
                    value = torch.zeros(shape, dtype=dtype)
                if not torch.is_tensor(value) or tuple(value.shape) != shape:
                    raise ValueError(
                        f"samples[{sample_index}] mano_parameters.{field} must have shape {shape}."
                    )
                if field == "valid" and value.dtype != torch.bool:
                    raise TypeError("mano_parameters.valid must be boolean.")
                values.append(value.to(device=device, dtype=dtype))
            mano_batch[field] = torch.stack(values, dim=0)
        mano_batch["side_order"] = ("right", "left")
        batch["mano_parameters"] = mano_batch

    if any(sample.get("objpose_records") is not None for sample in samples):
        batch["objpose_records"] = [
            sample.get("objpose_records", [None] * frame_count) for sample in samples
        ]
    for field, dtype in (
        ("objpose_frame_ids", torch.int64),
        ("objpose_record_present", torch.bool),
    ):
        if any(sample.get(field) is not None for sample in samples):
            values = []
            for sample_index, sample in enumerate(samples):
                value = sample.get(field)
                if value is None:
                    fill = -1 if dtype == torch.int64 else False
                    value = torch.full((frame_count,), fill, dtype=dtype)
                if not torch.is_tensor(value) or tuple(value.shape) != (frame_count,):
                    raise ValueError(f"samples[{sample_index}] {field} must have shape [T].")
                values.append(value.to(device=device, dtype=dtype))
            batch[field] = torch.stack(values, dim=0)

    if any(sample.get("action_segments") is not None for sample in samples):
        batch["action_segments"] = [
            [dict(item) for item in sample.get("action_segments", [])]
            for sample in samples
        ]
    if any(sample.get("metadata") is not None for sample in samples):
        batch["metadata"] = [dict(sample.get("metadata", {})) for sample in samples]

    # Derive geometry only when source depth and intrinsics were both collated.
    # Invalid/missing calibration remains invalid rather than being guessed.
    if "depth" in batch and "intrinsics" in batch:
        batch = HOI4DSupervisionBuilder()(batch)
    return batch


def make_hoi4d_dataloader(
    dataset: HOI4DDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool | None = None,
) -> DataLoader:
    """Create a DataLoader using the HOI4D-aware E3A collator."""
    if not isinstance(dataset, HOI4DDataset):
        raise TypeError("dataset must be an HOI4DDataset.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_hoi4d_batch,
    )
