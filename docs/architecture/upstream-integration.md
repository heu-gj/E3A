# Upstream integration contract

This document records the APIs verified at the pinned upstream revisions. If a submodule is updated, re-check every item and update the parity tests before changing E3A integration code.

## VGGT-Ω

Pinned upstream: facebookresearch/vggt-omega at 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6.

Verified upstream files:

- vggt_omega/models/vggt_omega.py
- vggt_omega/models/aggregator.py
- vggt_omega/models/heads/camera_head.py
- vggt_omega/models/heads/dense_head.py
- vggt_omega/utils/pose_enc.py
- training/losses/unprojection.py

### Existing upstream contracts

1. VGGTOmega.forward(images) accepts [T,3,H,W] or [B,T,3,H,W]; a 4D input is expanded to batch size one.
2. Aggregator.forward returns (aggregated_tokens_list, patch_token_start).
3. The cached Aggregator tensors concatenate per-frame and inter-frame representations on the feature dimension. With the released embed_dim=1024, cached tokens are 2048-D.
4. The original camera/register prefix length is 1 + num_register_tokens (17 for the released default).
5. CameraHead reads only the prefix supplied through patch_token_start.
6. DenseHead.forward returns (depth, depth_conf) by default. Its internal fused feature is at 1/4 input resolution and 256 channels for the released default configuration.
7. Official camera decoding is provided by encoding_to_camera(pose_enc, image_size_hw).
8. Official world-point unprojection is provided by training/losses/unprojection.py::unproject_depth_to_points_torch_batch.

### E3A compatibility patches

The pinned patches are stored in patches/vggt-omega/ and applied by scripts/apply_patches.sh.

- 0001-extra-tokens-and-dense-features.patch
  - Adds optional extra_tokens: [B,T,N_extra,1024] to Aggregator.forward.
  - Treats E3A tokens as prefix tokens for the register-only inter-frame attention blocks.
  - Returns the expanded prefix length for DenseHead patch slicing.
  - Adds DenseHead.forward(return_features=True), returning (depth, depth_conf, fused_features) with fused features shaped [B,T,256,H/4,W/4] for the released default configuration.
- 0002-official-forward-feature-api.patch
  - Extends the official VGGTOmega.forward with optional extra_tokens and return_features keyword arguments.
  - Keeps the no-extra-token model(images) output contract unchanged.
  - Keeps CameraHead and TextAlignmentHead on the original camera/register prefix.
  - Passes the expanded prefix only to DenseHead.
  - Exposes final E3A token features and, only when requested, the fused DenseHead feature.

The E3A adapter delegates to this patched official top-level forward. It does not duplicate VGGT-Ω's forward orchestration. No upstream parameter names or shapes are changed, so the original checkpoint is still loaded strictly.

### Required tests before accepting a patch update

- Default forward parity against the pinned unpatched forward contract.
- No-extra-token output keys and shapes remain unchanged.
- Original camera/register prefix and expanded E3A prefix have the expected lengths.
- Register-only inter-frame attention includes E3A tokens and excludes patch tokens, as in upstream.
- Dense fused features have the expected spatial size and are finite.
- Frame chunking preserves output order and frame count.
- Gradients reach E3A tokens and E3A heads when training is enabled.
- The original upstream checkpoint loads with strict=True, without missing/unexpected keys for original parameters.

### Geometry contract

Use encoding_to_camera from upstream for pose decoding. It returns camera-from-world extrinsics in the OpenCV convention. Use the official unprojection helper, which inverts those extrinsics and returns world-space points. The official helper uses pixel centers (u+0.5, v+0.5). Do not hand-roll a different camera parameter decoder or pixel-center convention.

depth_conf is not a calibrated probability in [0,1]; the official DenseHead emits 1 + exp(confidence_logits), and the official depth loss uses it as a learned positive weighting term. Do not multiply 3D pooling weights by depth_conf as though larger values were probabilities. Any use of this field must follow an explicitly validated calibration/uncertainty interpretation.

## SAM 3.1

Pinned upstream: facebookresearch/sam3 at 31cb7e624e4fe95301b1be1d4e809111593d2e45.

The pinned upstream recommends build_sam3_multiplex_video_predictor for SAM 3.1 multiplex video tracking. Its predictor exposes handle_request and handle_stream_request.

The E3A adapter in src/e3a/adapters/sam3_video.py is intentionally a thin request wrapper:

- start_session -> type="start_session"
- add_text_prompt -> type="add_prompt"
- propagate -> type="propagate_in_video"
- close_session -> type="close_session"

The adapter does not reinterpret masks or track IDs. Output tensor layout, object-ID semantics, and frame indexing must be inspected from actual upstream outputs and covered by conversion tests before teacher outputs enter E3A training.

### Reusing the SAM 3 MaskPredictor

The pinned upstream sam3/model/maskformer_segmentation.py::MaskPredictor accepts object queries [B,Q,C] and pixel embeddings [B,C,H,W]. It applies the upstream query MLP and computes query-pixel mask logits with an einsum, returning [B,Q,H,W].

EntityMaskHead reuses this exact class and adds only the feature-dimension projections needed to connect VGGT-Ω's 2048-D cached entity tokens and 256-D fused features to the SAM 3 mask dimension. The predictor instance is newly initialized for E3A; SAM 3 checkpoint weights are not assumed to transfer to VGGT-Ω features. This does not instantiate SAM 3's separate 848M-parameter vision-language detector/tracker in the online E3A path.

SAM 3.1's multiplex video predictor remains an optional offline teacher/baseline. It is a separate large model with its own visual encoder, tracker memory, preprocessing, session state, checkpoint, and latency/memory costs. Do not load it alongside VGGT-Ω in the default E3A forward pass.

SAM 3.1 checkpoints may require Hugging Face access approval/authentication. Do not commit checkpoints or access tokens.

## VidEoMT and EntitySAM

VidEoMT is MIT-licensed and is retained as an optional reference for video query propagation and mask training. Its matcher is a frame-batched video-segmentation matcher, not a drop-in replacement for E3A's clip-level identity matching; adapt only after validating E3A track-ID semantics.

EntitySAM is a useful method reference for automatic entity discovery, but its public repository has no repository-level LICENSE file. Do not copy or vendor its implementation until reuse permission is confirmed.
