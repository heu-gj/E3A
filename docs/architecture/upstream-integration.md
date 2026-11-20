# Upstream integration contract

This document records the APIs verified at the pinned upstream revisions. If a submodule is updated, re-check every item and update the parity tests before changing E3A integration code.

## VGGT-Ω

Pinned upstream: facebookresearch/vggt-omega at 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6

Verified files:

- vggt_omega/models/vggt_omega.py
- vggt_omega/models/aggregator.py
- vggt_omega/models/heads/camera_head.py
- vggt_omega/models/heads/dense_head.py
- vggt_omega/utils/pose_enc.py

### Existing interfaces

1. VGGTOmega.forward(images) accepts [T,3,H,W] or [B,T,3,H,W]; the former is expanded to a batch of one.
2. The top-level forward calls aggregator(images), then camera_head(..., patch_token_start=...) and dense_head(..., patch_token_start=...).
3. Aggregator.forward returns (aggregated_tokens_list, patch_token_start). Its cached layers are the configured indices (default: 4, 11, 17, 23).
4. DenseHead.forward currently returns exactly (depth, depth_conf). Its internal fused tensor is not part of the public output.
5. CameraHead attends over the prefix tokens passed through patch_token_start, then reads the first camera token.
6. Official camera decoding is provided by encoding_to_camera(pose_enc, image_size).

### Minimal patch plan (not yet applied)

When adding E3A Entity/Hand/Future tokens:

- Add an optional extra_tokens: [B,T,N_extra,D] argument to Aggregator.forward. Keep the default None path numerically and structurally identical to upstream.
- Preserve the original camera/register prefix length separately from the expanded prefix length. Pass the expanded prefix length to DenseHead so patch tokens are sliced correctly.
- Pass only the original camera/register prefix length to CameraHead. Otherwise the new E3A tokens would silently enter the camera-head token mixer.
- Pass the prefix length explicitly into the register-only inter-frame attention helper. Do not mutate self.patch_token_start per forward call; that would be unsafe for concurrent/re-entrant forwards.
- Add an opt-in return_features=False argument to DenseHead. When true, return the fused feature as a third output with explicit [B,T,C,Hf,Wf] shape. When false, preserve the original two-output API exactly.
- Do not alter existing parameter names or shapes. New E3A token parameters belong to E3A, not the upstream checkpoint.
- Load the original VGGT-Ω checkpoint strictly into the unmodified upstream model before adding E3A-only parameters, or demonstrate strict loading after the minimal patch. No silent strict=False checkpoint loading.

### Required tests before accepting the patch

- No-extra-token parity: same model weights and inputs produce the same camera, depth, and confidence outputs as the unpatched pinned upstream.
- Prefix indexing: original camera/register prefix and expanded E3A prefix have the expected lengths.
- Register-attention path: E3A tokens participate in register-only cross-frame attention; patch tokens remain excluded there, as in upstream.
- Dense feature contract: fused features have expected spatial size and are finite.
- Gradient contract: gradients reach E3A tokens and E3A heads when training is enabled.
- Checkpoint contract: original upstream checkpoint loads without missing/unexpected keys for all original parameters.

### Geometry contract

Use encoding_to_camera from upstream for pose decoding. Verify the returned extrinsics convention from the pinned implementation and use the same convention in unprojection. Do not hand-roll a different camera parameter decoder.

## SAM 3.1

Pinned upstream: facebookresearch/sam3 at 31cb7e624e4fe95301b1be1d4e809111593d2e45

The current upstream recommends build_sam3_multiplex_video_predictor for SAM 3.1 multiplex video tracking. It returns a predictor with handle_request and handle_stream_request APIs.

The E3A adapter in src/e3a/adapters/sam3_video.py is intentionally a thin request wrapper:

- start_session -> type=start_session
- add_text_prompt -> type=add_prompt
- propagate -> type=propagate_in_video
- close_session -> type=close_session

The adapter does not reinterpret masks or track IDs. Output tensor layout, object-ID semantics, and frame indexing must be inspected from the returned upstream outputs and covered by dataset-specific conversion tests before they enter E3A training.

SAM 3.1 checkpoint downloads may require Hugging Face access approval/authentication. Do not commit checkpoints.

## VidEoMT and EntitySAM

VidEoMT is an MIT-licensed reference implementation for video query propagation and mask training. Its current README also points to newer PMT/LVMT work; benchmark current alternatives before adopting its architecture.

EntitySAM is a useful method reference for automatic entity discovery, but its public repository has no repository-level LICENSE file. Do not copy its implementation until licensing permission is confirmed.
