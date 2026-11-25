"""Run a local HOI4D Dataset/DataLoader smoke test without training the model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from e3a.data.hoi4d_dataset import (
    HOI4DDataset,
    make_hoi4d_dataloader,
    resolve_hoi4d_sequence_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("HOI4D_ROOT"))
    parser.add_argument(
        "--sequence",
        default="ZY20210800001/H1/C20/N30/S295/s04/T4",
    )
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-mano", action="store_true")
    parser.add_argument("--no-objpose", action="store_true")
    args = parser.parse_args()
    if not args.root:
        parser.error("Provide --root or set HOI4D_ROOT.")
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be provided together.")
    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    image_size = (
        (args.height, args.width)
        if args.height is not None and args.width is not None
        else None
    )
    dataset = HOI4DDataset(
        root=Path(args.root),
        sequence_ids=args.sequence,
        clip_length=args.clip_length,
        frame_stride=1,
        clip_stride=args.clip_length,
        image_size=image_size,
        load_depth=not args.no_depth,
        load_mano=not args.no_mano,
        load_objpose=not args.no_objpose,
    )
    loader = make_hoi4d_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    paths = resolve_hoi4d_sequence_paths(args.root, args.sequence)
    capture = cv2.VideoCapture(str(paths.rgb_video))
    try:
        expected_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()

    seen_frames: set[int] = set()
    batches = 0
    entity_valid_clips = 0
    invalid_mask_pixels = 0
    mano_valid_frames = 0
    objpose_records = 0
    depth_valid_pixels = 0
    geometry_valid_track_frames = 0
    geometry_metric_valid_clips = 0
    action_segments = 0
    action_valid_clips = 0
    time_valid_clips = 0
    first_batch_shapes = {}

    for batch in loader:
        batches += 1
        frame_indices = batch["frame_indices"][0].tolist()
        seen_frames.update(int(index) for index in frame_indices)
        entity_valid_clips += int(batch["entity_supervision_valid"].sum().item())
        invalid_mask_pixels += int(
            (~batch["entity_targets"]["pixel_valid"]).sum().item()
        )
        if "mano_parameters" in batch:
            mano_valid_frames += int(batch["mano_parameters"]["valid"].sum().item())
        if "objpose_record_present" in batch:
            objpose_records += int(batch["objpose_record_present"].sum().item())
        if "depth_valid" in batch:
            depth_valid_pixels += int(batch["depth_valid"].sum().item())
        if "geometry_targets" in batch:
            geometry_valid_track_frames += int(
                batch["geometry_targets"]["valid"].sum().item()
            )
            geometry_metric_valid_clips += int(
                batch["geometry_targets"]["metric_valid"].sum().item()
            )
        if batches == 1:
            action_segments = len(batch.get("action_segments", [[]])[0])
        action_valid_clips += int(
            batch["anticipation_targets"]["action_valid"].sum().item()
        )
        time_valid_clips += int(
            batch["anticipation_targets"]["time_valid"].sum().item()
        )
        if batches == 1:
            first_batch_shapes = {
                "images": list(batch["images"].shape),
                "depth": list(batch["depth"].shape) if "depth" in batch else None,
                "entity_masks": list(batch["entity_targets"]["masks"].shape),
                "pixel_valid": list(batch["entity_targets"]["pixel_valid"].shape),
                "intrinsics": list(batch["intrinsics"].shape) if "intrinsics" in batch else None,
                "geometry_centers_camera": (
                    list(batch["geometry_targets"]["centers_camera"].shape)
                    if "geometry_targets" in batch else None
                ),
                "geometry_covariance_camera": (
                    list(batch["geometry_targets"]["covariance_camera"].shape)
                    if "geometry_targets" in batch else None
                ),
                "geometry_valid": (
                    list(batch["geometry_targets"]["valid"].shape)
                    if "geometry_targets" in batch else None
                ),
            }
        if args.max_batches and batches >= args.max_batches:
            break

    complete_scan = args.max_batches == 0
    summary = {
        "sequence_id": args.sequence,
        "num_dataset_clips": len(dataset),
        "batches_read": batches,
        "expected_video_frames": expected_frames,
        "unique_source_frames_seen": len(seen_frames),
        "all_source_frames_covered": (
            seen_frames == set(range(expected_frames)) if complete_scan else None
        ),
        "entity_supervised_clips": entity_valid_clips,
        "invalid_or_unannotated_mask_pixels": invalid_mask_pixels,
        "mano_valid_hand_frames": mano_valid_frames,
        "objpose_records_read": objpose_records,
        "depth_valid_pixels": depth_valid_pixels,
        "geometry_valid_track_frames": geometry_valid_track_frames,
        "geometry_metric_valid_clips": geometry_metric_valid_clips,
        "action_segments_loaded": action_segments,
        "action_supervised_clips": action_valid_clips,
        "time_supervised_clips": time_valid_clips,
        "first_batch_shapes": first_batch_shapes,
        "physical_units_and_objpose_coordinate_frame": "remain unknown",
        "model_forward_or_backward": "not run",
    }
    if complete_scan and seen_frames != set(range(expected_frames)):
        raise RuntimeError(
            "The DataLoader scan did not cover every source video frame: "
            f"{len(seen_frames)}/{expected_frames}."
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
