# E3A architecture

## Research objective

Given an observed video prefix, predict the future interaction target, action, and event time. The model must not consume frames or annotations after the observation cutoff.

The architecture is organized around one geometry-aware video representation, not several independent visual backbones.

## Responsibility split

| Responsibility | Implementation source |
|---|---|
| Multi-frame geometry features, camera pose, dense depth | Official VGGT-Ω |
| Optional video mask generation and tracking baseline/teacher | Official SAM 3.1 |
| Query-propagation research baseline | Official VidEoMT |
| Entity mask prediction | E3A feature adapter plus the official SAM 3 MaskPredictor |
| Entity-to-future-target association | E3A-owned task head and matching/loss code |
| 3D human-object relative state | E3A geometry adapter using official VGGT-Ω outputs |
| Future action and event-time prediction | E3A-owned anticipation head |
| Dataset semantics, event windows, split policy | E3A-owned audited data pipeline |

## Important architectural choices

- VGGT-Ω is the primary online visual path.
- SAM 3.1 is an optional offline teacher / segmentation-tracking baseline. Do not load its second large vision encoder in the default E3A forward pass.
- E3A reuses SAM 3's official MaskPredictor query-to-mask module, not the full SAM 3 detector/tracker. Its weights are newly initialized for VGGT-Ω features; SAM 3 checkpoint transfer is not assumed.
- E3A entity slots are clip-local slots, not permanent global object IDs. Clip-level matching is valid only after the dataset adapter supplies consistent track IDs and explicit disappearance/occlusion semantics.
- The number of entity slots, object classes, hand joints, and action classes must be selected from dataset statistics and annotation audits, not copied from illustrative code.

## VGGT-Ω feature path

The official Aggregator produces cached intermediate token tensors. The E3A compatibility patch inserts per-frame entity/hand/future tokens as extra prefix tokens and exposes the DenseHead's fused 1/4-resolution feature. The patched top-level VGGTOmega.forward owns the forward path so E3A does not duplicate upstream camera/depth orchestration.

E3ARepresentationModel currently composes:
1. E3AEncoder: per-frame entity/hand/future token injection into VGGT-Ω.
2. EntityMaskHead: SAM 3's official MaskPredictor with E3A-owned dimension projections.
3. Official VGGT-Ω camera decoding and depth unprojection.
4. Differentiable soft-mask pooling of world-space points into entity centers and covariance.

Depth confidence is deliberately not treated as a probability; the official VGGT-Ω loss uses it as a learned positive weighting term.

See docs/architecture/upstream-integration.md for pinned API details, patch contracts, coordinate conventions, and required tests.

## Current scope

The repository now has a testable entity/3D representation path. It is not yet a complete/trainable action-anticipation model: dataset/event-label semantics, tube-level identity matching, hand supervision, and future target/action/time losses remain to be implemented after data auditing.
