"""Ego-Exo4D EgoPose and object-relation annotation adapters.

Schemas are taken from the official Ego-Exo4D annotation documentation and
baseline readers. EgoPose 3D joints are world-frame points; the official
baseline applies the per-frame camera extrinsics as p_cam = [R|t] p_world.
Relation masks are LZ-string-compressed COCO RLE and annotated_frames is the
source of annotation coverage (absence from the mask dictionary is not a
negative label unless the frame is listed there).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e3a.data.actions import action_label_key


EGOEXO_HAND_ORDER = ("right", "left")
EGOEXO_HAND_JOINTS = (
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_4",
    "index_1",
    "index_2",
    "index_3",
    "index_4",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_4",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_4",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_4",
)


def _load_json_mapping(path: str | Path, *, description: str) -> Mapping[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} JSON root must be an object.")
    return value


def _resolve_camera_extrinsics(
    camera_pose: Mapping[str, Any],
    frame_number: int,
    camera_name: str,
) -> np.ndarray:
    """Support both published EgoPose camera-pose JSON layouts."""
    frame_key = str(frame_number)
    candidate: Any = None

    # Current documentation example: frame -> camera -> camera_extrinsics.
    frame_record = camera_pose.get(frame_key)
    if isinstance(frame_record, Mapping):
        camera_record = frame_record.get(camera_name)
        if isinstance(camera_record, Mapping):
            candidate = camera_record.get("camera_extrinsics")

    # Official baseline reader layout: camera -> camera_extrinsics -> frame.
    if candidate is None:
        camera_record = camera_pose.get(camera_name)
        if isinstance(camera_record, Mapping):
            extrinsics_by_frame = camera_record.get("camera_extrinsics")
            if isinstance(extrinsics_by_frame, Mapping):
                candidate = extrinsics_by_frame.get(frame_key)
            elif extrinsics_by_frame is not None:
                candidate = extrinsics_by_frame

    if candidate is None:
        raise KeyError(
            f"Missing camera_extrinsics for camera={camera_name!r}, frame={frame_number}."
        )

    extrinsics = np.asarray(candidate, dtype=np.float64)
    if extrinsics.shape != (3, 4) or not np.isfinite(extrinsics).all():
        raise ValueError(
            f"camera_extrinsics for {camera_name}/{frame_number} must be finite [3,4]."
        )
    return extrinsics


def parse_egoexo_hand_pose_annotations(
    annotations: Mapping[str, Any],
    camera_pose: Mapping[str, Any],
    *,
    camera_name: str = "aria",
    min_views_for_3d: int | None,
    source_to_model_units: float = 1.0,
    strict_camera_pose: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    """Convert EgoPose world-frame hand joints to per-frame camera coordinates.

    Output frames contain:
      keypoints_camera: float32 [2,21,3], in right-then-left order
      valid: bool [2,21]

    The 21-joint order matches the official EgoPose baseline's finger_dict:
    wrist, thumb(1..4), index(1..4), middle(1..4), ring(1..4), pinky(1..4).
    Coordinates are multiplied by source_to_model_units after camera
    transformation. Its default 1.0 preserves source units; it does not assert
    that the source unit is meters. min_views_for_3d is required so the caller
    must choose the manual/automatic annotation policy explicitly. Set it to
    None for the official
    automatic-annotation path, which does not apply the manual >=3-view filter.
    """
    if min_views_for_3d is not None and min_views_for_3d < 0:
        raise ValueError("min_views_for_3d must be non-negative or None.")
    if not np.isfinite(source_to_model_units) or source_to_model_units <= 0:
        raise ValueError("source_to_model_units must be finite and positive.")
    if not isinstance(annotations, Mapping) or not isinstance(camera_pose, Mapping):
        raise TypeError("annotations and camera_pose must be JSON object mappings.")

    output: dict[int, dict[str, torch.Tensor]] = {}
    for frame_key, frame_records in annotations.items():
        try:
            frame_number = int(frame_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"EgoPose frame key {frame_key!r} is not an integer.") from exc
        if not isinstance(frame_records, Sequence) or isinstance(frame_records, (str, bytes)):
            raise TypeError(f"EgoPose frame {frame_number} must contain a record list.")
        if len(frame_records) == 0:
            continue
        frame_record = frame_records[0]
        if not isinstance(frame_record, Mapping):
            raise TypeError(f"EgoPose frame {frame_number} record must be an object.")
        annotation_3d = frame_record.get("annotation3D")
        if not isinstance(annotation_3d, Mapping) or len(annotation_3d) == 0:
            continue

        try:
            extrinsics = _resolve_camera_extrinsics(
                camera_pose,
                frame_number,
                camera_name,
            )
        except KeyError:
            if strict_camera_pose:
                raise
            continue

        keypoints_world = np.zeros((2, 21, 3), dtype=np.float64)
        valid = np.zeros((2, 21), dtype=np.bool_)
        for hand_index, hand_name in enumerate(EGOEXO_HAND_ORDER):
            for joint_index, joint_name in enumerate(EGOEXO_HAND_JOINTS):
                joint = annotation_3d.get(f"{hand_name}_{joint_name}")
                if not isinstance(joint, Mapping):
                    continue
                try:
                    point = np.asarray(
                        [joint["x"], joint["y"], joint["z"]],
                        dtype=np.float64,
                    )
                    view_count = (
                        int(joint["num_views_for_3d"])
                        if min_views_for_3d is not None
                        else None
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    (min_views_for_3d is not None and view_count < min_views_for_3d)
                    or not np.isfinite(point).all()
                ):
                    continue
                keypoints_world[hand_index, joint_index] = point
                valid[hand_index, joint_index] = True

        rotation = extrinsics[:, :3]
        translation = extrinsics[:, 3]
        keypoints_camera = (
            keypoints_world @ rotation.T + translation[None, None, :]
        ) * float(source_to_model_units)
        keypoints_camera[~valid] = 0.0

        output[frame_number] = {
            "keypoints_camera": torch.from_numpy(keypoints_camera.astype(np.float32)),
            "valid": torch.from_numpy(valid),
        }
    return output


def load_egoexo_hand_pose_annotations(
    annotation_path: str | Path,
    camera_pose_path: str | Path,
    **kwargs: Any,
) -> dict[int, dict[str, torch.Tensor]]:
    """Load the official per-take EgoPose hand JSON and camera-pose JSON files."""
    annotations = _load_json_mapping(annotation_path, description="EgoPose hand annotations")
    camera_pose = _load_json_mapping(camera_pose_path, description="EgoPose camera pose")
    return parse_egoexo_hand_pose_annotations(annotations, camera_pose, **kwargs)


def decode_egoexo_relation_mask(mask_record: Mapping[str, Any]) -> np.ndarray:
    """Decode one official encodedMask record using the Ego4D SDK decoder."""
    if not isinstance(mask_record, Mapping):
        raise TypeError("Ego-Exo4D encoded mask record must be an object.")
    required = {"width", "height", "encodedMask"}
    missing = required.difference(mask_record)
    if missing:
        raise ValueError(f"Ego-Exo4D mask record is missing {sorted(missing)}.")
    try:
        from ego4d.research.util.masks import decode_mask
    except ImportError as exc:
        raise ImportError(
            "Decoding Ego-Exo4D relation masks requires the official Ego4D SDK "
            "and pycocotools. Install the Ego4D package and pycocotools."
        ) from exc

    decoded = np.asarray(decode_mask(dict(mask_record)))
    expected = (int(mask_record["height"]), int(mask_record["width"]))
    if decoded.shape != expected:
        raise ValueError(
            f"Decoded Ego-Exo4D mask has shape {decoded.shape}, expected {expected}."
        )
    return decoded.astype(np.bool_, copy=False)


def build_egoexo_relation_clip_targets(
    relation_annotations: Mapping[str, Any],
    take_uid: str,
    camera_name: str,
    frame_numbers: Sequence[int],
    *,
    image_size: tuple[int, int] | None = None,
    mask_decoder: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
    exclude_hand_tracks: bool = True,
) -> dict[str, Any]:
    """Build VidEoMT-compatible object tracks for one take/camera/frame window.

    For each object and selected frame:
      - frame in annotated_frames + encoded mask: positive object mask;
      - frame in annotated_frames + no encoded mask: explicit annotated absence;
      - frame not in annotated_frames: unannotated (mask_valid=False).

    Tracks that are never visible inside this clip are omitted because the
    clip-consistent matcher requires a visible observation to establish a slot.
    Object IDs are deterministic clip-local integers; track_names preserves the
    source object-name mapping for downstream audit.
    """
    if (
        not isinstance(frame_numbers, Sequence)
        or isinstance(frame_numbers, (str, bytes))
        or len(frame_numbers) == 0
    ):
        raise ValueError("frame_numbers must be a non-empty sequence of frame indices.")
    frames = [int(value) for value in frame_numbers]
    if any(right <= left for left, right in zip(frames, frames[1:])):
        raise ValueError("frame_numbers must be strictly increasing.")
    if image_size is not None:
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must be a positive (height,width) tuple.")
        image_size = (int(image_size[0]), int(image_size[1]))

    root = relation_annotations.get("annotations", relation_annotations)
    if not isinstance(root, Mapping) or take_uid not in root:
        raise KeyError(f"Take {take_uid!r} is missing from Ego-Exo4D relation annotations.")
    take_record = root[take_uid]
    if not isinstance(take_record, Mapping):
        raise TypeError(f"Relation annotation for take {take_uid!r} must be an object.")
    object_masks = take_record.get("object_masks")
    if not isinstance(object_masks, Mapping):
        raise ValueError(f"Take {take_uid!r} has no object_masks mapping.")

    decoder = mask_decoder or decode_egoexo_relation_mask
    prepared: list[dict[str, Any]] = []
    coverage_sets: list[set[int]] = []

    # object_names enumerates active objects, while object_masks can be a
    # sampled subset. Coverage is only complete when the two non-hand track
    # sets agree; otherwise unmatched queries are not safe no-object negatives.
    object_names_record = take_record.get("object_names")
    enumerated_names: set[str] = set()
    enumeration_complete = False
    if isinstance(object_names_record, Mapping):
        enumerated_annotations = object_names_record.get("annotation")
        if isinstance(enumerated_annotations, Sequence) and not isinstance(
            enumerated_annotations, (str, bytes)
        ):
            enumeration_complete = True
            for annotation_record in enumerated_annotations:
                if not isinstance(annotation_record, Mapping):
                    enumeration_complete = False
                    break
                track_name = annotation_record.get("track_id")
                if not isinstance(track_name, str) or not track_name:
                    enumeration_complete = False
                    break
                normalized_track_name = re.sub(
                    r"[_\s]+", " ", track_name.lower()
                ).strip()
                is_hand_track = re.fullmatch(
                    r"(?:(?:left|right) )?hand(?: \d+)?",
                    normalized_track_name,
                ) is not None
                if not (exclude_hand_tracks and is_hand_track):
                    enumerated_names.add(track_name)

    mask_names = set()
    for object_name in object_masks:
        normalized_name = re.sub(
            r"[_\s]+", " ", str(object_name).lower()
        ).strip()
        is_hand_track = re.fullmatch(
            r"(?:(?:left|right) )?hand(?: \d+)?",
            normalized_name,
        ) is not None
        if not (exclude_hand_tracks and is_hand_track):
            mask_names.add(str(object_name))
    coverage_complete = enumeration_complete and enumerated_names == mask_names
    inferred_size = image_size

    for object_name in sorted(object_masks):
        normalized_name = re.sub(r"[_\s]+", " ", str(object_name).lower()).strip()
        is_hand_track = re.fullmatch(
            r"(?:(?:left|right) )?hand(?: \d+)?",
            normalized_name,
        ) is not None
        if exclude_hand_tracks and is_hand_track:
            continue
        cameras = object_masks[object_name]
        if not isinstance(cameras, Mapping) or camera_name not in cameras:
            coverage_complete = False
            continue
        camera_record = cameras[camera_name]
        if not isinstance(camera_record, Mapping):
            raise TypeError(
                f"Relation camera record for {object_name}/{camera_name} must be an object."
            )

        annotated_frames_raw = camera_record.get("annotated_frames")
        annotation = camera_record.get("annotation")
        if not isinstance(annotated_frames_raw, Sequence) or isinstance(
            annotated_frames_raw, (str, bytes)
        ):
            raise ValueError(
                f"Relation track {object_name!r} is missing annotated_frames coverage."
            )
        if not isinstance(annotation, Mapping):
            raise ValueError(f"Relation track {object_name!r} is missing its annotation mapping.")
        annotated_frames = {int(value) for value in annotated_frames_raw}
        coverage_sets.append(annotated_frames)
        annotation_frames = {int(value) for value in annotation}
        if not annotation_frames.issubset(annotated_frames):
            raise ValueError(
                f"Relation track {object_name!r} contains masks outside annotated_frames."
            )

        first_record = next(iter(annotation.values()), None)
        if first_record is not None:
            if not isinstance(first_record, Mapping):
                raise TypeError(f"Mask record for {object_name!r} must be an object.")
            record_size = (int(first_record["height"]), int(first_record["width"]))
            if inferred_size is None:
                inferred_size = record_size
            elif record_size != inferred_size:
                raise ValueError(
                    f"Mask size for {object_name!r} is {record_size}, expected {inferred_size}."
                )

        selected_masks: dict[int, np.ndarray] = {}
        visible_in_window = False
        for frame_number in frames:
            record = annotation.get(str(frame_number), annotation.get(frame_number))
            if record is None:
                continue
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Mask record for {object_name!r} frame {frame_number} must be an object."
                )
            current_size = (int(record["height"]), int(record["width"]))
            if inferred_size is None:
                inferred_size = current_size
            elif current_size != inferred_size:
                raise ValueError(
                    f"Mask size for {object_name!r}/{frame_number} is {current_size}, "
                    f"expected {inferred_size}."
                )
            decoded = np.asarray(decoder(record))
            if decoded.shape != current_size:
                raise ValueError(
                    f"Decoded mask for {object_name!r}/{frame_number} has shape "
                    f"{decoded.shape}, expected {current_size}."
                )
            binary = decoded.astype(np.bool_, copy=False)
            if not binary.any():
                raise ValueError(
                    f"Encoded mask for {object_name!r}/{frame_number} is empty; "
                    "the source schema uses missing mask entries for annotated absence."
                )
            selected_masks[frame_number] = binary
            visible_in_window = True

        if visible_in_window:
            prepared.append(
                {
                    "name": str(object_name),
                    "annotated_frames": annotated_frames,
                    "masks": selected_masks,
                }
            )

    if inferred_size is None:
        raise ValueError(
            "Cannot infer Ego-Exo4D mask dimensions; pass image_size for this camera."
        )
    height, width = inferred_size
    track_count = len(prepared)
    frame_count = len(frames)
    labels = torch.zeros(track_count, dtype=torch.int64)
    ids = torch.full((track_count, frame_count), -1, dtype=torch.int64)
    masks = torch.zeros((track_count, frame_count, height, width), dtype=torch.float32)
    mask_valid = torch.zeros((track_count, frame_count), dtype=torch.bool)

    for track_index, track in enumerate(prepared):
        for frame_index, frame_number in enumerate(frames):
            is_annotated = frame_number in track["annotated_frames"]
            mask_valid[track_index, frame_index] = is_annotated
            binary = track["masks"].get(frame_number)
            if binary is not None:
                ids[track_index, frame_index] = track_index
                masks[track_index, frame_index] = torch.from_numpy(
                    binary.astype(np.float32)
                )

    frame_supervision_valid = torch.tensor(
        [
            bool(coverage_complete and coverage_sets)
            and all(frame_number in coverage for coverage in coverage_sets)
            for frame_number in frames
        ],
        dtype=torch.bool,
    )
    entity_supervision_valid = bool(
        (track_count > 0 and mask_valid.any().item())
        or (track_count == 0 and frame_supervision_valid.any().item())
    )
    return {
        "labels": labels,
        "ids": ids,
        "masks": masks,
        "mask_valid": mask_valid,
        "entity_supervision_valid": entity_supervision_valid,
        "entity_frame_supervision_valid": frame_supervision_valid,
        "track_names": [track["name"] for track in prepared],
        "frame_numbers": frames,
    }


def audit_egoexo_relation_coverage(
    relation_annotations: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit metadata required for safe Ego-Exo4D no-object supervision.

    This is read-only and does not decode masks. A camera is eligible only
    when object_names and object_masks enumerate the same non-hand tracks and
    every enumerated object has a valid annotated_frames list for that camera.
    """
    root = relation_annotations.get("annotations", relation_annotations)
    if not isinstance(root, Mapping):
        raise TypeError("Ego-Exo4D relation annotations root must be an object.")

    take_reports: list[dict[str, Any]] = []
    for take_uid, take_record in root.items():
        if not isinstance(take_record, Mapping):
            take_reports.append(
                {
                    "take_uid": str(take_uid),
                    "status": "INVALID_TAKE_RECORD",
                    "object_names_valid": False,
                    "object_masks_valid": False,
                    "enumeration_matches": False,
                    "enumerated_nonhand_tracks": [],
                    "mask_nonhand_tracks": [],
                    "enumerated_only_tracks": [],
                    "mask_only_tracks": [],
                    "cameras": [],
                }
            )
            continue

        raw_object_masks = take_record.get("object_masks")
        object_masks_valid = isinstance(raw_object_masks, Mapping)
        object_masks = raw_object_masks if object_masks_valid else {}

        object_names_record = take_record.get("object_names")
        enumerated_names: set[str] = set()
        enumeration_complete = False
        if isinstance(object_names_record, Mapping):
            annotations = object_names_record.get("annotation")
            if isinstance(annotations, Sequence) and not isinstance(
                annotations, (str, bytes)
            ):
                enumeration_complete = True
                for annotation_record in annotations:
                    if not isinstance(annotation_record, Mapping):
                        enumeration_complete = False
                        break
                    track_name = annotation_record.get("track_id")
                    if not isinstance(track_name, str) or not track_name:
                        enumeration_complete = False
                        break
                    normalized = re.sub(r"[_\s]+", " ", track_name.lower()).strip()
                    if re.fullmatch(r"(?:(?:left|right) )?hand(?: \d+)?", normalized):
                        continue
                    enumerated_names.add(track_name)

        mask_names: set[str] = set()
        for object_name in object_masks:
            normalized = re.sub(
                r"[_\s]+", " ", str(object_name).lower()
            ).strip()
            if not re.fullmatch(r"(?:(?:left|right) )?hand(?: \d+)?", normalized):
                mask_names.add(str(object_name))

        enumeration_matches = bool(
            object_masks_valid
            and enumeration_complete
            and enumerated_names == mask_names
        )
        camera_names = sorted(
            {
                str(camera_name)
                for track_name in mask_names
                if isinstance(object_masks.get(track_name), Mapping)
                for camera_name in object_masks[track_name]
            }
        )

        camera_reports: list[dict[str, Any]] = []
        for camera_name in camera_names:
            missing_tracks: list[str] = []
            invalid_tracks: list[str] = []
            coverage_sets: list[set[int]] = []
            for track_name in sorted(enumerated_names):
                cameras = object_masks.get(track_name)
                if not isinstance(cameras, Mapping) or camera_name not in cameras:
                    missing_tracks.append(track_name)
                    continue
                camera_record = cameras[camera_name]
                if not isinstance(camera_record, Mapping):
                    invalid_tracks.append(track_name)
                    continue
                raw_frames = camera_record.get("annotated_frames")
                if not isinstance(raw_frames, Sequence) or isinstance(
                    raw_frames, (str, bytes)
                ):
                    invalid_tracks.append(track_name)
                    continue
                try:
                    coverage_sets.append({int(frame) for frame in raw_frames})
                except (TypeError, ValueError):
                    invalid_tracks.append(track_name)

            all_tracks_covered = bool(
                enumerated_names
                and not missing_tracks
                and not invalid_tracks
                and len(coverage_sets) == len(enumerated_names)
            )
            common_frames = (
                sorted(set.intersection(*coverage_sets))
                if all_tracks_covered
                else []
            )
            coverage_complete = bool(enumeration_matches and all_tracks_covered)
            camera_reports.append(
                {
                    "camera_name": camera_name,
                    "enumerated_track_count": len(enumerated_names),
                    "camera_track_coverage_count": len(coverage_sets),
                    "missing_camera_tracks": missing_tracks,
                    "invalid_coverage_tracks": invalid_tracks,
                    "coverage_complete": coverage_complete,
                    "eligible_for_no_object": bool(
                        coverage_complete and common_frames
                    ),
                    "common_annotated_frames": common_frames,
                    "common_annotated_frame_count": len(common_frames),
                }
            )

        eligible_camera_count = sum(
            bool(camera["eligible_for_no_object"]) for camera in camera_reports
        )
        if not object_masks_valid:
            status = "MISSING_OBJECT_MASKS"
        elif not enumeration_complete:
            status = "INVALID_OBJECT_NAMES"
        elif not enumeration_matches:
            status = "OBJECT_ENUMERATION_MISMATCH"
        elif eligible_camera_count:
            status = "ELIGIBLE_CAMERA_FOUND"
        else:
            status = "NO_COMPLETE_CAMERA_COVERAGE"

        take_reports.append(
            {
                "take_uid": str(take_uid),
                "status": status,
                "object_names_valid": enumeration_complete,
                "object_masks_valid": object_masks_valid,
                "enumeration_matches": enumeration_matches,
                "enumerated_nonhand_tracks": sorted(enumerated_names),
                "mask_nonhand_tracks": sorted(mask_names),
                "enumerated_only_tracks": sorted(enumerated_names - mask_names),
                "mask_only_tracks": sorted(mask_names - enumerated_names),
                "camera_count": len(camera_reports),
                "eligible_camera_count": eligible_camera_count,
                "eligible_camera_frame_pairs": sum(
                    camera["common_annotated_frame_count"]
                    for camera in camera_reports
                    if camera["eligible_for_no_object"]
                ),
                "cameras": camera_reports,
            }
        )

    all_cameras = [
        camera for take in take_reports for camera in take.get("cameras", [])
    ]
    summary = {
        "take_count": len(take_reports),
        "takes_with_valid_object_names": sum(
            bool(take.get("object_names_valid")) for take in take_reports
        ),
        "takes_with_matching_object_enumeration": sum(
            bool(take.get("enumeration_matches")) for take in take_reports
        ),
        "takes_with_any_eligible_camera": sum(
            bool(take.get("eligible_camera_count", 0)) for take in take_reports
        ),
        "camera_count": len(all_cameras),
        "cameras_with_complete_coverage": sum(
            bool(camera["coverage_complete"]) for camera in all_cameras
        ),
        "eligible_camera_count": sum(
            bool(camera["eligible_for_no_object"]) for camera in all_cameras
        ),
        "eligible_camera_frame_pairs": sum(
            int(camera["common_annotated_frame_count"])
            for camera in all_cameras
            if camera["eligible_for_no_object"]
        ),
    }
    return {"summary": summary, "takes": take_reports}


def parse_egoexo_keystep_segments(
    annotations: Mapping[str, Any],
    take_uid: str,
) -> list[dict[str, Any]]:
    """Read the documented Ego-Exo4D keystep segments for one take.

    The returned source labels are step_unique_id values, not scenario-local
    step_id values and not the fine-grained benchmark's label_id values.
    """
    root = annotations.get("annotations", annotations)
    if not isinstance(root, Mapping) or take_uid not in root:
        raise KeyError(f"Take {take_uid!r} is missing from Ego-Exo4D keystep annotations.")
    take_record = root[take_uid]
    if not isinstance(take_record, Mapping):
        raise TypeError(f"Keystep annotation for take {take_uid!r} must be an object.")
    raw_segments = take_record.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise ValueError(f"Keystep annotation for take {take_uid!r} has no segments list.")

    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise TypeError(f"Keystep segment {index} for take {take_uid!r} must be an object.")
        try:
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            step_unique_id = int(segment["step_unique_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Keystep segment {index} for take {take_uid!r} has invalid time/step fields."
            ) from exc
        if (
            not np.isfinite(start_time)
            or not np.isfinite(end_time)
            or start_time < 0
            or end_time <= start_time
            or step_unique_id < 0
        ):
            raise ValueError(
                f"Keystep segment {index} for take {take_uid!r} has invalid interval or ID."
            )
        segments.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "step_unique_id": step_unique_id,
                "step_name": str(segment.get("step_name", "")),
            }
        )
    segments.sort(key=lambda item: (item["start_time"], item["end_time"]))
    return segments


def load_egoexo_keystep_segments(
    annotation_path: str | Path,
    take_uid: str,
) -> list[dict[str, Any]]:
    """Load a raw Ego-Exo4D keystep annotation JSON for one take."""
    annotations = _load_json_mapping(annotation_path, description="Ego-Exo4D keystep annotations")
    return parse_egoexo_keystep_segments(annotations, take_uid)


def build_egoexo_keystep_vocabulary(
    training_annotations: Mapping[str, Any],
) -> dict[int, int]:
    """Build a deterministic action-index map from training split step_unique_id labels."""
    root = training_annotations.get("annotations", training_annotations)
    if not isinstance(root, Mapping):
        raise TypeError("Keystep training annotations must contain an annotations mapping.")
    step_ids: set[int] = set()
    for take_uid in sorted(root):
        take_record = root[take_uid]
        if not isinstance(take_record, Mapping):
            raise TypeError(f"Keystep annotation for take {take_uid!r} must be an object.")
        segments = take_record.get("segments")
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            raise ValueError(f"Keystep annotation for take {take_uid!r} has no segments list.")
        for segment in segments:
            if not isinstance(segment, Mapping) or "step_unique_id" not in segment:
                raise ValueError(f"Keystep annotation for take {take_uid!r} has an invalid segment.")
            step_id = int(segment["step_unique_id"])
            if step_id < 0:
                raise ValueError("step_unique_id values must be non-negative.")
            step_ids.add(step_id)
    return {step_id: index for index, step_id in enumerate(sorted(step_ids))}


def build_egoexo_anticipation_target(
    segments: Sequence[Mapping[str, Any]],
    observation_end_time: float,
    action_vocabulary: Mapping[Any, int],
) -> dict[str, Any]:
    """Create future action/time labels without exposing future frames as inputs.

    The first keystep whose start_time is strictly after the observation cutoff
    defines the event. A currently active/overlapping segment is not treated as
    a future event. Unknown validation/test step IDs keep time supervision but
    disable action classification. Keystep annotations do not identify the
    target object, so target-entity supervision remains invalid.
    """
    cutoff = float(observation_end_time)
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("observation_end_time must be finite and non-negative.")

    future_segments: list[tuple[float, float, int]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise TypeError(f"segments[{index}] must be an object.")
        try:
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            step_id = int(segment["step_unique_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"segments[{index}] has invalid start_time/end_time/step_unique_id."
            ) from exc
        if (
            not np.isfinite(start_time)
            or not np.isfinite(end_time)
            or start_time < 0
            or end_time <= start_time
            or step_id < 0
        ):
            raise ValueError(f"segments[{index}] has invalid interval or step ID.")
        if start_time > cutoff:
            future_segments.append((start_time, end_time, step_id))

    if not future_segments:
        return {
            "target_entity_track_id": -1,
            "target_entity_valid": False,
            "action_label": -1,
            "action_valid": False,
            "time_to_event": 0.0,
            "time_valid": False,
        }

    event_start, _, step_id = min(future_segments, key=lambda item: (item[0], item[1], item[2]))
    prefixed_key = action_label_key("egoexo4d", f"step_unique_id={step_id}")
    if prefixed_key in action_vocabulary:
        raw_action_label = action_vocabulary[prefixed_key]
        action_valid = True
    elif step_id in action_vocabulary:
        # Compatibility for an Ego-Exo4D-only vocabulary.
        raw_action_label = action_vocabulary[step_id]
        action_valid = True
    else:
        raw_action_label = -1
        action_valid = False
    if action_valid:
        if isinstance(raw_action_label, bool) or not isinstance(raw_action_label, (int, np.integer)):
            raise TypeError("Ego-Exo4D action vocabulary indices must be integers.")
        action_label = int(raw_action_label)
        if action_label < 0:
            raise ValueError("Ego-Exo4D action vocabulary indices must be non-negative.")
    else:
        action_label = -1
    return {
        "target_entity_track_id": -1,
        "target_entity_valid": False,
        "action_label": action_label,
        "action_valid": action_valid,
        "time_to_event": float(event_start - cutoff),
        "time_valid": True,
    }


def resolve_egoexo_video_path(
    release_root: str | Path,
    takes: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    take_uid: str,
    camera_name: str,
    *,
    require_exists: bool = True,
) -> Path:
    """Resolve a frame-aligned video using the official Ego4D notebook convention."""
    if isinstance(takes, Mapping):
        take_record = takes.get(take_uid)
    else:
        take_record = next(
            (take for take in takes if take.get("take_uid") == take_uid),
            None,
        )
    if not isinstance(take_record, Mapping):
        raise KeyError(f"take_uid {take_uid!r} is missing from takes metadata.")

    parts = camera_name.split("_")
    if len(parts) == 1:
        camera_id, stream_id = parts[0], "0"
    elif len(parts) == 2:
        camera_id, stream_id = parts
    else:
        raise ValueError(
            f"Unexpected Ego-Exo4D camera name {camera_name!r}; expected camNN or camNN_stream."
        )
    if stream_id == "214-1":
        stream_id = "rgb"

    videos = take_record.get("frame_aligned_videos")
    if not isinstance(videos, Mapping):
        raise ValueError(f"Take {take_uid!r} has no frame_aligned_videos mapping.")
    camera_streams = videos.get(camera_id)
    if not isinstance(camera_streams, Mapping) or stream_id not in camera_streams:
        raise KeyError(
            f"Missing frame-aligned stream {camera_id}/{stream_id} for take {take_uid!r}."
        )
    relative_path = camera_streams[stream_id].get("relative_path")
    root_dir = take_record.get("root_dir")
    if not isinstance(relative_path, str) or not isinstance(root_dir, str):
        raise ValueError(f"Take {take_uid!r} has invalid root_dir/relative_path metadata.")

    path = Path(release_root).expanduser() / root_dir / relative_path
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"Ego-Exo4D video does not exist: {path}")
    return path
