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
| Entity mask prediction | E3A dimension adapter plus the official SAM 3 MaskPredictor implementation |\n| Entity-to-future-target association | E3A-owned task head and matching/loss code |
| 3D human-object relative state | E3A-owned geometry adapter using VGGT-Ω outputs |
| Future action and event-time prediction | E3A-owned anticipation head |
| Dataset semantics, event windows, split policy | E3A-owned audited data pipeline |

## Important architectural choice

SAM 3.1 and VGGT-Ω each contain a large vision encoder. The initial E3A online path must not load both by default. Use SAM 3.1 offline to create a baseline, inspect annotations, or generate teacher outputs where allowed. The E3A model's primary visual path remains VGGT-Ω.

The E3A EntityMaskHead reuses SAM 3's official MaskPredictor class, with a small projection from VGGT-Ω entity features to the mask embedding dimension. This reuses the upstream query-to-mask implementation; it does not claim that SAM 3 checkpoint weights transfer to VGGT-Ω features.\n\nSAM 3.1 is not a drop-in patch to VGGT-Ω. Its predictor has its own detector, tracker, memory, video-session lifecycle, image preprocessing, and checkpoint. If later experiments justify joint use, it must be introduced as an explicit two-model variant with separate latency and memory accounting.

## VGGT-Ω integration boundary

The official VGGTOmega.forward(images) is the source of truth for the released model API. At the pinned revision, its documented inference outputs include:

- camera_and_register_tokens
- pose_enc
- depth
- depth_conf
- images in evaluation mode
- text_alignment_embedding only when alignment is enabled

The top-level forward does not return a point_map or a fused dense feature map. Do not invent either key.

The official Aggregator.forward returns cached intermediate token tensors and patch_token_start. The official DenseHead uses these tensors to form a fused 1/4-resolution feature internally, but its public forward currently returns depth and confidence only. A minimal opt-in feature-return patch may be needed for E3A mask/query heads. That patch must preserve the default output contract and be tested against the pinned upstream revision.

## 3D geometry

Decode pose_enc with the official encoding_to_camera utility. Use the official camera convention and depth convention when unprojecting pixels. Document whether points are expressed in camera or reconstruction/world coordinates, and test transforms with known synthetic points. Do not treat pose encodings as already-decoded extrinsics.

## Entity identity

Do not assume that an entity slot is a permanent global object ID. The initial identity contract is clip-local: one matched slot corresponds to one GT track within a sampled video clip. Clip-level matching is valid only if the dataset adapter supplies consistent track IDs across the clip and handles object appearance/disappearance explicitly.

The number of entity slots, object classes, hand joints, and action classes must be selected from dataset statistics and annotation audits—not hard-coded from an example architecture.
