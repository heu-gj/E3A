# HOI4D Dataset and DataLoader

## Minimal use

The raw reader is implemented in e3a.data.hoi4d_dataset. It consumes the
locally extracted release layout audited in
reports/audits/P1-HOI4D-Physical-Data-Audit/ and does not modify source files.

    import os

    from e3a.data import HOI4DDataset, make_hoi4d_dataloader

    train_dataset = HOI4DDataset(
        root=os.environ["HOI4D_ROOT"],
        sequence_ids=[
            "ZY20210800001/H1/C20/N30/S295/s04/T4",
        ],
        clip_length=8,
        frame_stride=1,
        clip_stride=8,
        image_size=None,
        action_vocabulary=train_only_action_vocabulary,
    )

    train_loader = make_hoi4d_dataloader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    batch = next(iter(train_loader))

sequence_ids must be supplied explicitly. Build train/validation/test lists at
the sequence or subject level before constructing datasets; do not split
overlapping clips from one source video across partitions.

## Sample contract

Each item contains:

- images: RGB float32 [T,3,H,W] in [0,1].
- timestamps: float32 [T], calculated as decoded video frame index divided
  by the measured RGB FPS. These are video-timeline timestamps, not sensor
  capture timestamps.
- frame_indices: original decoded RGB/depth/mask filename indices [T].
- depth: float32 [T,1,H,W] retaining raw depth values; no unit conversion.
- depth_valid: bool [T,1,H,W], true for finite positive source values.
- intrinsics: camera matrix [3,3], scaled only for the optional image resize.
- entity_targets: existing labels, ids, masks, mask_valid, plus
  pixel_valid [T,H,W].
- entity_supervision_valid: false for clips without visible object-track
  supervision. Such clips must not silently become all-no-object examples.
- mano_parameters: raw pose coefficients, shape, translation, 2D keypoints,
  and frame-level validity in right-then-left order. This is not a reconstructed
  3D hand-pose target.
- objpose_records: raw per-frame JSON records. objpose_frame_ids preserves the
  source frameId field separately from frame_indices.
- action_segments and anticipation_targets: source action intervals and
  strictly-future action/time targets. Target-entity validity remains false.
- metadata: explicit unknown physical units/coordinate-frame declarations
  and the provisional status of category-local mask instance IDs.

The HOI4D-aware collator delegates the canonical E3A tensors to
collate_e3a_batch and additionally batches depth, intrinsics, frame indices,
MANO raw parameters, pixel validity, and clip-level entity validity.

## Data-integrity rules

1. The source RGB and depth streams are decoded by frame index. The depth
   reader sets CAP_PROP_CONVERT_RGB=0; depth is not normalized or divided by
   1000 because its physical unit is not yet verified.
2. RGB/depth metadata (frame count, FPS, and dimensions) must agree by default.
   This is not a claim of pixel-perfect calibration or sensor timestamp sync.
3. Mask PNGs are converted from palette mode to RGB before decoding. Unknown
   colors are assigned semantic/instance -1 and pixel_valid=False. The
   temporary background value used to call the existing strict target builder
   is excluded by pixel_valid; it is not negative supervision.
4. mask_valid remains track/frame-level and is true only when the track has
   visible labeled pixels. pixel_valid separately excludes unknown pixels.
   entity_supervision_valid=False for a clip with no visible entity track.
5. objpose files are addressed by their numeric filename stem. The raw
   frameId is retained as a separate field and is never used as a filename or
   video index. Pose coordinates, units, and transform direction remain
   unknown.
6. MANO pickles use a restricted NumPy-array-only unpickler. Missing hand frames
   remain invalid. Raw MANO parameters and 2D keypoints are loaded, but no MANO
   forward pass, joint reordering, 3D reconstruction, or unit conversion is
   performed.
7. Action labels are looked up only in the vocabulary passed by the caller.
   Construct that vocabulary from the training split only. The source action
   intervals do not identify the target entity, so target-entity supervision is
   always invalid.

## H1 validation

After installing the E3A package in the project environment, run the real-root
smoke test. It scans every source frame by default using non-overlapping clips
plus a final end-aligned clip:

    HOI4D_ROOT=/path/to/HOI4D python scripts/smoke_test_hoi4d_dataset.py

For a lower-memory smoke test, request a consistent image-space resize:

    HOI4D_ROOT=/path/to/HOI4D python scripts/smoke_test_hoi4d_dataset.py --height 540 --width 960

Use --max-batches 1 to test only the first batch. The full scan reports source
frame coverage, tensor shapes, mask validity, MANO/objpose coverage, depth
validity, and action/time target availability. It does not run the model or
claim physical GT validation.

Unit tests:

Run:

    python -m pytest -q tests/test_hoi4d_dataset.py tests/test_hoi4d_data.py tests/test_e3a_collate.py

A synthetic integration fixture exercises video decoding contracts, mask
validity, raw pose preservation, MANO pickle restrictions, action/time targets,
and the HOI4D-aware collator. It does not replace the real-root smoke test on
the locally downloaded HOI4D release.
