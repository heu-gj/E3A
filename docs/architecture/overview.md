# E3A v0.2 architecture

## Research objective

Given an observed RGB video prefix, predict the future interaction target entity, action, and event time. The model must not consume frames or annotations after the observation cutoff.

The design follows the VGGT philosophy:

**One Encoder · One Entity Space · Multiple Predictions**

One VGGT-Ω geometry/video backbone receives E3A's extra tokens. All task heads read the shared representation and execute in one forward pass. SAM 3.1 is not part of the online path.

## Unified computation graph

```text
Observed RGB clip + timestamps
              |
       VGGT-Ω Aggregator
              |
  Frame/Patch + Scene Registers
              |
   E3A extra registers (per frame)
   - Entity Registers (K slots)
   - Hand Registers (2 slots)
   - Future Register (1 slot)
              |
      Shared geometry-aware
       spatiotemporal features
              |
       Parallel task heads
   +----------+-----------+-----------+-----------+
   |          |           |           |           |
Geometry   Entity       Hand      Interaction  Anticipation
Head       Mask Head    Pose Head Head         Head
   |          |           |           |           |
Depth/     Masks +     3D hand    Contact +   Target slot +
camera +   objectness  joints     relative    action +
3D points                         geometry    time-to-event
```

## Shared representation and identity

E3ATokenBank injects K entity slots, two hand slots, and one future slot into the official VGGT-Ω Aggregator. The token bank input width is the Aggregator embedding width; the exposed cached entity/hand/future features use the Aggregator's concatenated two-stream width (2048-D for the pinned default VGGT-Ω configuration).

The same entity slot index is present at every observed frame. During training, VidEoMT's VideoHungarianMatcher_Consistent assigns a GT track to a query at its first visible frame and reuses that assignment across the clip. E3A exposes this exact GT-track-to-query mapping from the upstream criterion and reuses it for interaction and target-entity supervision.

Slots are **clip-local**, not permanent global object IDs. If a target object is not visible or cannot be matched in the observed clip, its target-entity label must be marked invalid; the current head does not invent an unseen entity slot.

PersistentEntityPool performs learned temporal pooling over each slot's per-frame features. It creates one clip-level entity state for anticipation without adding a separate tracking network or a recurrent filter.

## Parallel prediction heads

| Head | Input | Output |
|---|---|---|
| GeometryHead | VGGT-Ω depth and camera encoding + entity masks | World points, camera matrices, per-frame entity center/covariance |
| EntityMaskHead | Per-frame entity query + fused DenseHead feature | Query-to-pixel mask logits and objectness |
| HandPoseHead | Two hand registers | Two camera-frame hands, 21 joints each, plus joint visibility logits |
| InteractionHead | Hand/entity tokens + 3D relative state | Pairwise interaction embeddings and contact logits |
| AnticipationHead | Future register + persistent entity/interaction states | Target slot, action class, log-normal event-time parameters |

The native EntityMaskHead projects queries and dense pixels into a shared mask space and computes mask logits by query-pixel dot product. It does not load SAM 3 or a second visual backbone.

Hand joint order is canonicalized as wrist, thumb, index, middle, ring, pinky (four joints per finger). Dataset adapters must convert source annotations into this order and camera coordinate frame before enabling hand supervision.

Interaction features use hand-wrist-to-entity-center relative position, finite-difference relative velocity, distance, and projected hand/entity features. This is a lightweight pairwise MLP, not a separate graph transformer.

The AnticipationHead predicts:
- target_entity_logits: class 0 means explicitly annotated no-target; classes 1..K point to the corresponding clip-local entity slot.
- action_logits: a dataset-audited action vocabulary. HOI4D and Ego-Exo4D labels must use a stable union vocabulary with dataset-prefixed IDs unless an audited semantic mapping proves classes equivalent.
- time_log_mean and time_log_scale: parameters of a log-normal time-to-event distribution in seconds.

## Unified training

E3AMultiTaskLoss combines:
- VidEoMT entity class/mask losses,
- masked 3D hand keypoint regression and optional visibility loss,
- matched-slot hand-object contact loss,
- target-entity classification using the matcher-provided track-to-slot map,
- action classification,
- log-normal time-to-event negative log-likelihood.

Every task uses an explicit validity mask. Missing annotations are ignored, not converted into negative labels. The loss returns one loss_objective for VGGT-Ω's existing Trainer, so there is one optimizer and one joint backward pass.

collate_e3a_batch pads variable entity tracks while keeping all targets batch-first for the upstream gradient-accumulation chunker.

## Dataset responsibilities

- **HOI4D:** RGB-D, motion/panoptic segmentation, object pose, MANO hand annotations, and action intervals. The official motion masks are palette-coded and must be decoded with the official category/instance mapping.
- **Ego-Exo4D:** object mask tracks, 3D hand/body pose, synchronized views, and time-indexed keysteps. annotated_frames must be distinguished from frames with an actual encoded mask.
- Raw action vocabularies are not interchangeable. Dataset adapters must produce a documented global label map and validity masks.
- The current implementation includes the model and loss contracts; raw dataset readers, source-to-canonical coordinate transforms, event-window sampling, and leak-free split generation are still being implemented and audited.

## Upstream responsibilities

| Responsibility | Source |
|---|---|
| Video/geometry backbone and camera/depth prediction | Official VGGT-Ω |
| Clip-consistent query assignment and point-sampled mask criterion | Official VidEoMT |
| Optional independent teacher/baseline | Official SAM 3.1; never loaded by default E3A forward |
| Task heads, persistent clip entity state, joint loss, dataset contracts | E3A |

Depth confidence is not treated as a calibrated probability. The official VGGT-Ω DenseHead emits a learned positive weighting term; E3A's geometric pooling uses finite points and valid positive depth.

See docs/architecture/upstream-integration.md and docs/implementation/training.md for exact interfaces.
