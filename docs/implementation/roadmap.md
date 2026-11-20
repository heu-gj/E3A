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
- [ ] Add visual inspection and round-trip tests for samples.

Acceptance: a deterministic dataset sample can be traced back to original annotations, with all units and coordinate conventions documented.

## P2 — VGGT-Ω feature integration

- [x] Add the minimal optional extra-token path to the pinned Aggregator (patch committed; CI validation pending).
- [x] Preserve the original camera/register prefix for CameraHead in the E3A feature adapter.
- [x] Add opt-in DenseHead fused-feature return while preserving default outputs (patch committed; CI validation pending).
- [x] Add E3A token bank and unit shape/gradient tests.
- [ ] Run no-extra-token parity and checkpoint compatibility tests.

Acceptance: pretrained geometry outputs remain unchanged when E3A tokens are absent; E3A tokens receive gradients when enabled.

## P3 — Entity masks and 3D entity state

- [ ] Decide whether initial supervision comes from dataset GT masks, SAM 3.1 teacher outputs, or both.
- [ ] Implement only the task-specific mask/query adapter that upstream code does not already provide.
- [ ] Implement clip-level identity matching only after track-ID semantics are validated.
- [ ] Decode camera parameters with upstream utilities and unproject depth using the verified convention.
- [ ] Add tests for empty masks, partial visibility, occlusion, invalid depth, and coordinate transforms.

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
