# Training framework reuse

E3A reuses the VGGT-Ω training implementation in third_party/vggt-omega/training:

- Trainer: DDP/FSDP setup, gradient accumulation, AMP/scaling, optimizer/scheduler construction, logging, checkpoint/resume, and the training loop.
- train_utils: distributed setup, checkpoint helpers, optimizer construction, gradient clipping, and device transfer.
- Hydra: component construction from E3A-owned configs.

E3A does not fork the training loop. E3ATrainer overrides only task-specific assumptions in the upstream Trainer:

1. Pass the batch timestamps to the E3A model along with RGB frames.
2. Derive batch size from the RGB tensor instead of requiring batch["extrinsics"].
3. Wrap a predicted pose_enc in pose_enc_list, matching the official VGGT-Ω MultitaskLoss interface when camera supervision is retained as an auxiliary objective.

For entity-mask supervision, use e3a.adapters.videomt.VidEoMTMaskCriterion. It accepts the same loss call signature as the upstream Trainer: loss(predictions, batch, schedule_progress). The batch must contain entity_targets, a list with one clip-target dictionary per video.

Each clip-target dictionary uses:

- labels [N]: int64, all zero for the one foreground category "entity"
- ids [N,T]: int64, stable non-negative track ID while visible and -1 when not visible
- masks [N,T,H,W]: binary/soft masks, zero wherever ids == -1

The adapter wraps the pinned upstream VideoHungarianMatcher_Consistent and VideoSetCriterion, converts the clip-level target schema to VidEoMT's B*T frame-wise contract, and returns loss_objective for the upstream optimizer loop. It requires the number of visible tracks in each clip not to exceed the configured E3A entity slots. It rejects unstable/duplicate track IDs and invalid mask/visibility combinations. This is an adapter contract, not a claim that any particular dataset already conforms to it.

The E3A dataset must return at least images [B,T,3,H,W], timestamps [B,T], entity_targets, and any additional fields required by the selected criterion. The criterion must return loss_objective, as expected by the upstream optimizer loop.

Install the upstream training extra:

    python -m pip install -e "third_party/vggt-omega[train]"

VidEoMT additionally requires compatible Detectron2 and SciPy dependencies. Install/validate those in the target PyTorch/CUDA environment before enabling the real matcher/criterion.

The actual training config is intentionally deferred until the dataset sample schema, event labels, and loss contract are audited. Do not create a runnable-looking config with guessed label names or dimensions.
