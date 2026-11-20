# E3A implementation roadmap

A milestone is complete only when its acceptance tests pass and the evidence is recorded. A directory existing or a forward pass returning tensors is not sufficient.

## P0 — Reproducible upstream baseline

- [x] Pin VGGT-Ω, SAM 3, and VidEoMT as Git submodules.
- [x] Add thin wrappers around official VGGT-Ω and SAM 3.1 APIs.
- [ ] Create a clean environment using the pinned upstreams.
- [ ] Load an approved VGGT-Ω checkpoint with strict state-dict loading.
- [ ] Run official and adapter inference on the same frames; compare every output key and tensor.
- [ ] Record GPU model, CUDA/PyTorch versions, input resolution, peak memory, and latency.

Acceptance: adapter output parity with official VGGT-Ω; no E3A-specific model head is claimed yet.

## P1 — Dataset and label contract

- [ ] Audit source video frame rates, timestamps, image/depth alignment, camera metadata, instance masks, track IDs, hand annotations, action labels, and event boundaries.
- [ ] Define a clip sample schema with explicit coordinate frames and units.
- [ ] Define observation cutoff and future prediction windows without future-frame leakage.
- [ ] Define missing/occluded/disappeared-object semantics.
- [ ] Define train/validation/test splits at the source-video/subject level to prevent leakage.
- [ ] Add visual inspection and round-trip tests.

Acceptance: a deterministic dataset sample can be traced back to original annotations, with all units and coordinate conventions documented.

## P2 — VGGT-Ω feature integration

- [x] Add the minimal optional extra-token path to the pinned Aggregator.
- [x] Preserve the original camera/register prefix for CameraHead in the E3A feature adapter.
- [x] Add opt-in DenseHead fused-feature return while preserving default outputs.
- [x] Delegate the optional E3A feature path through the official top-level VGGTOmega.forward.
- [x] Add E3A token bank and unit shape/gradient tests.
- [x] Apply the patches to a clean pinned submodule in CI and pass patch-application tests.
- [x] Pass CPU unit tests for Aggregator prefix behavior, token gradients, top-level output contract, and DenseHead feature chunking.
- [ ] Run no-extra-token parity and strict checkpoint compatibility with an approved full VGGT-Ω checkpoint.
- [ ] Record GPU runtime and peak memory.

Acceptance: pretrained geometry outputs remain unchanged when E3A tokens are absent; E3A tokens receive gradients when enabled.

## P3 — Entity masks and 3D entity state

- [x] Add the E3A dimension adapter around SAM 3's official MaskPredictor.
- [x] Decode camera parameters with official VGGT-Ω utilities and unproject depth with the official helper.
- [x] Add differentiable soft-mask pooling for world-space entity centers and covariance.
- [x] Add a thin adapter to VidEoMT's official VideoHungarianMatcher_Consistent and VideoSetCriterion, including a fail-closed clip target contract.
- [x] Add CPU tests for VidEoMT frame-wise input conversion, loss aggregation, empty clips, track-ID validation, and batch-first target collation.
- [x] Verify padded entity targets remain aligned when the pinned VGGT-Ω gradient-accumulation chunker splits a batch.
- [ ] Validate the real SAM 3 MaskPredictor import and forward pass in the target GPU environment.
- [ ] Validate the real VidEoMT matcher/criterion with compatible Detectron2/SciPy dependencies.
- [ ] Confirm the dataset's track IDs, visibility, and masks satisfy the adapter contract.
- [ ] Decide whether initial supervision comes from dataset GT masks, SAM 3.1 teacher outputs, or both.
- [ ] Add dataset-backed tests for partial visibility, occlusion, invalid depth, and coordinate transforms.

Acceptance: per-clip entity masks/IDs and 3D states can be evaluated against held-out annotations.

## P4 — Human-object interaction representation

- [ ] Select hand representation from actual available annotations (2D, 3D, keypoints, or masks).
- [ ] Define hand-object relative geometry with explicit units and coordinate frame.
- [ ] Add missing-hand and low-confidence handling.
- [ ] Verify that geometry features improve a controlled baseline.

Acceptance: hand/object relation features are finite, physically interpretable, and evaluated on held-out data.

## P5 — Future target/action/time prediction

- [ ] Define action taxonomy and label mapping from the dataset.
- [ ] Define future target identity mapping to clip-local entity IDs.
- [ ] Define event onset and time-to-event semantics.
- [ ] Implement target, action, and time losses with class imbalance and missing-label handling.
- [ ] Add no-future-leakage tests and baseline comparisons.

Acceptance: the model predicts future target/action/time from observed frames only, with valid metrics and reproducible splits.

## P6 — Joint training and paper experiments

- [ ] Start with frozen VGGT-Ω and train only E3A-owned modules.
- [ ] Unfreeze selected upstream blocks only after stable baselines.
- [ ] Compare RGB-only, geometry-only, entity-only, and full E3A variants.
- [ ] Compare SAM 3.1 as an independent segmentation/tracking baseline.
- [ ] Report accuracy, calibration, runtime, memory, and failure cases.

Acceptance: every claimed gain is supported by controlled experiments and ablations.
