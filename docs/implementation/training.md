# Training framework reuse

E3A reuses the VGGT-Ω training implementation in third_party/vggt-omega/training:

- Trainer: DDP/FSDP setup, gradient accumulation, AMP/scaling, optimizer/scheduler construction, logging, checkpoint/resume, and the training loop.
- train_utils: distributed setup, checkpoint helpers, optimizer construction, gradient clipping, and device transfer.
- Hydra: component construction from E3A-owned configs.

E3A does not fork the training loop. E3ATrainer overrides only two task-specific assumptions in the upstream Trainer:

1. Pass the batch timestamps to the E3A model along with RGB frames.
2. Derive batch size from the RGB tensor instead of requiring batch["extrinsics"].

The E3A dataset must return at least images [B,T,3,H,W], timestamps [B,T], and the fields required by the configured E3A criterion. The criterion must return loss_objective, as expected by the upstream optimizer loop.

Install the upstream training extra:

    python -m pip install -e "third_party/vggt-omega[train]"

The actual training config is intentionally deferred until the dataset sample schema, event labels, and loss contract are audited. Do not create a runnable-looking config with guessed label names or dimensions.
