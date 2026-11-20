# E3A implementation roadmap

A milestone is complete only when its acceptance tests pass and the evidence is recorded. A directory existing or a forward pass returning tensors is not sufficient.

## P0 — Reproducible upstream baseline

- [x] Pin VGGT-Ω, SAM 3, and VidEoMT as Git submodules.
- [x] Add thin official-API adapters.
- [ ] Create a clean environment using the pinned upstreams.
- [ ] Load an approved VGGT-Ω checkpoint with strict state-dict loading.
- [ ] Run official and adapter inference on the same frames; compare every output key and tensor.
- [ ] Record GPU model, CUDA/PyTorch versions, input resolution, peak memory, and latency.

Acceptance: adapter output parity with official VGGT-Ω; no E3A-specific model head is claimed yet.

## P1 — HOI4D + Ego-Exo4D data and label contract

- [x] Verify official HOI4D data organization, action interval fields, palette-coded 2D motion masks, category/instance maps, object pose fields, and MANO parameter fields.
- [x] Verify Ego-Exo4D relation mask JSON fields, encoded-mask representation, annotated_frames semantics, EgoPose 3D hand-joint schema, and Keystep segment schema.
- [x] Implement the official HOI4D RGB palette decoder and clip-level object-track conversion.
- [x] Implement Ego-Exo4D EgoPose world-to-camera hand-joint parsing with both published camera-pose JSON layouts.
- [x] Implement Ego-Exo4D relation mask parsing with annotated_frames-aware validity.
- [ ] Audit the locally downloaded release versions, video FPS/frame indexing, image/depth alignment, and annotation coverage.
- [ ] Implement end-to-end source-video Dataset/DataLoader classes for both datasets.
- [ ] Audit source-to-canonical coordinate transforms and physical units on real sequences.
- [ ] Build a dataset-prefixed union action vocabulary; do not merge classes based on name similarity alone.
- [ ] Define observation cutoffs, future event windows, target-entity mapping, and time-to-event semantics without future-frame leakage.
- [ ] Define missing/occluded/disappeared-object semantics across all source annotations.
- [ ] Create source-video/subject-level train/validation/test splits to prevent leakage.

Acceptance: a deterministic sample from each dataset can be traced to source annotations with verified units, coordinate frames, track IDs, event boundaries, and annotation-validity semantics.

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

## P3 — Persistent entity masks and 3D state

- [x] Replace the online SAM 3 MaskPredictor dependency with a native query-to-pixel EntityMaskHead.
- [x] Decode camera parameters with official VGGT-Ω utilities and unproject depth with the official helper.
- [x] Add differentiable soft-mask pooling for per-frame 3D entity centers and covariance.
- [x] Add learned temporal pooling to create clip-level persistent entity states.
- [x] Reuse VidEoMT's official clip-consistent matcher and criterion.
- [x] Expose the upstream matcher assignment as GT-track-ID to E3A-slot mapping for downstream target/interaction supervision.
- [x] Add CPU tests for native mask decoding, gradients, temporal pooling, matcher assignments, and empty clips.
- [ ] Validate source dataset track IDs, visibility, and masks against the adapter contract.
- [ ] Add dataset-backed tests for partial visibility, occlusion, invalid depth, and coordinate transforms.
- [ ] Run the full model with an approved VGGT-Ω checkpoint on target GPU hardware.

Acceptance: held-out entity masks/IDs and 3D states can be evaluated on both datasets.

## P4 — Human-object interaction representation

- [x] Add a two-hand, 21-joint camera-frame HandPoseHead.
- [x] Convert predicted hand keypoints to world coordinates using VGGT-Ω camera extrinsics.
- [x] Add a lightweight pairwise InteractionHead using relative position, velocity, distance, and shared entity/hand features.
- [x] Add masked hand-pose and matched-slot contact losses.
- [ ] Convert HOI4D MANO and Ego-Exo4D EgoPose annotations into the canonical joint order and camera coordinate frame.
- [ ] Verify physical units and coordinate transforms on sampled real sequences.
- [ ] Evaluate hand pose and contact on held-out data.

Acceptance: hand/object relation features are finite, physically interpretable, and evaluated on held-out data.

## P5 — Future target/action/time prediction

- [x] Add an AnticipationHead that predicts target entity slot, action class, and log-normal time-to-event parameters.
- [x] Use the matcher-provided GT-track-to-slot mapping for target-entity supervision.
- [x] Add masked target-entity, action, and time-to-event losses.
- [ ] Generate the dataset-prefixed union action vocabulary from audited source labels.
- [ ] Define future event onset and target-entity mapping for HOI4D and Ego-Exo4D separately.
- [ ] Ensure all target labels are derived only from future annotations and all model inputs stop at the observation cutoff.
- [ ] Add no-future-leakage tests and anticipation baselines.

Acceptance: the model predicts future target/action/time from observed frames only, with valid metrics and reproducible splits.

## P6 — Joint training and paper experiments

- [x] Add E3AMultiTaskLoss with one loss_objective for the upstream VGGT-Ω Trainer.
- [x] Add batch-first E3A collation for entity, hand, interaction, and anticipation targets.
- [ ] Add deterministic source-video sampling and complete dataset readers.
- [ ] Create the runnable Hydra training config after the label vocabulary and dataset schema are frozen.
- [ ] Start with frozen VGGT-Ω and train only E3A-owned modules.
- [ ] Unfreeze selected upstream blocks only after stable baselines.
- [ ] Compare RGB-only, geometry-only, entity-only, and full E3A variants.
- [ ] Compare SAM 3.1 as an independent segmentation/tracking baseline.
- [ ] Report accuracy, calibration, runtime, memory, and failure cases.

Acceptance: every claimed gain is supported by controlled experiments and ablations.
