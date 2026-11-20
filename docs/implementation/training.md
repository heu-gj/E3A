# Training framework reuse

E3A reuses the VGGT-Ω training implementation in third_party/vggt-omega/training:

- Trainer: DDP/FSDP setup, gradient accumulation, AMP/scaling, optimizer/scheduler construction, logging, checkpoint/resume, and the training loop.
- train_utils: distributed setup, checkpoint helpers, optimizer construction, gradient clipping, and device transfer.
- Hydra: component construction from E3A-owned configs.

E3A does not fork the training loop. E3ATrainer overrides only task-specific assumptions in the upstream Trainer:

1. Pass the batch timestamps to the E3A model along with RGB frames.
2. Derive batch size from the RGB tensor instead of requiring batch["extrinsics"].
3. Wrap a predicted pose_enc in pose_enc_list, matching the official VGGT-Ω MultitaskLoss interface when camera supervision is retained as an auxiliary objective.

For entity-mask supervision, use e3a.adapters.videomt.VidEoMTMaskCriterion. It accepts the same loss call signature as the upstream Trainer: loss(predictions, batch, schedule_progress). The batch must contain entity_targets as a padded tensor mapping so the upstream recursive gradient-accumulation chunker splits targets along the video batch dimension rather than along the entity dimension.

The entity_targets mapping uses:

- labels [B,Nmax]: int64, zero for real and padded rows (the one foreground category is "entity")
- ids [B,Nmax,T]: int64, stable non-negative track ID while visible, -1 for missing frames and padded rows
- masks [B,Nmax,T,H,W]: binary/soft masks, zero wherever ids == -1 and for padded rows
- track_counts [B]: int64 number of real track rows in each clip

Each unpadded clip target uses labels [N], ids [N,T], and masks [N,T,H,W]. The adapter unwraps the padded mapping, validates stable/unique track IDs and visibility/mask semantics, then converts the clip-level target schema to VidEoMT's B*T frame-wise contract. It wraps the pinned upstream VideoHungarianMatcher_Consistent and VideoSetCriterion and returns loss_objective for the upstream optimizer loop. It requires the number of visible tracks in each clip not to exceed the configured E3A entity slots. This is an adapter contract, not a claim that any particular dataset already conforms to it.

Use e3a.data.collate.collate_entity_targets(per_clip_targets) to create the padded entity_targets mapping from variable-length per-clip annotations. The E3A dataset must return at least images [B,T,3,H,W], timestamps [B,T], entity_targets, and any additional fields required by the selected criterion. The padded entity_targets tensors must all have B as their first dimension so the upstream gradient-accumulation chunker keeps them aligned. The criterion must return loss_objective, as expected by the upstream optimizer loop.

Install the upstream training extra:

    python -m pip install -e "third_party/vggt-omega[train]"

VidEoMT additionally requires compatible Detectron2 and SciPy dependencies. Install/validate those in the target PyTorch/CUDA environment before enabling the real matcher/criterion.

The actual training config is intentionally deferred until the dataset sample schema, event labels, and loss contract are audited. Do not create a runnable-looking config with guessed label names or dimensions.
