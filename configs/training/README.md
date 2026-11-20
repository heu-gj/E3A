# Training configuration

The E3A training configuration is intentionally not populated yet.

Before creating e3a.yaml, complete the dataset/label audit and specify:

- dataset root and deterministic train/validation split,
- frame sampling and timestamp units,
- action taxonomy and event-onset semantics,
- target entity-ID mapping,
- mask/hand/geometry supervision availability,
- optimizer, AMP, checkpoint, and distributed settings.

The launcher reuses the pinned VGGT-Ω Trainer through E3ATrainer. Do not copy the upstream training loop into E3A.
