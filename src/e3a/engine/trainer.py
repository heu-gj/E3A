"""E3A-specific bridge to VGGT-Ω's official DDP/FSDP Trainer.

Only the model-input and batch-size assumptions are overridden. Optimizer
construction, AMP/scaling, gradient accumulation, distributed setup, logging,
checkpoint/resume, and the training loop remain upstream-owned.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_TRAINING = _REPO_ROOT / "third_party" / "vggt-omega" / "training"
if not _UPSTREAM_TRAINING.is_dir():
    raise ImportError(
        "VGGT-Ω training source is missing. Run "
        "git submodule update --init third_party/vggt-omega."
    )
if str(_UPSTREAM_TRAINING) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_TRAINING))

try:
    from trainer import Trainer as VGGTTrainer
    from train_utils.general import get_amp_type
except ImportError as exc:
    raise ImportError(
        "VGGT-Ω training dependencies are missing. Install the upstream train extra with "
        "pip install -e 'third_party/vggt-omega[train]'."
    ) from exc


class E3ATrainer(VGGTTrainer):
    """Reuse the upstream Trainer while passing timestamps to E3A."""

    def _step(
        self,
        batch: Mapping,
        model: torch.nn.Module,
        phase: str,
    ):
        with torch.autocast(
            device_type="cuda",
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            predictions = model(
                batch["images"],
                timestamps=batch["timestamps"],
            )

        # Preserve the official VGGT-Ω MultitaskLoss interface when E3A
        # experiments choose to retain camera supervision as an auxiliary loss.
        if "pose_enc" in predictions:
            predictions["pose_enc_list"] = [predictions["pose_enc"]]

        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = self.loss(predictions, batch, self.schedule_progress)

        log_data = {**predictions, **loss_dict, **batch}
        scalar_values, batch_size = self._collect_scalar_values(log_data, phase)
        return loss_dict, scalar_values, batch_size

    def _collect_scalar_values(
        self,
        batch: Mapping,
        phase: str,
    ) -> tuple[dict[str, float], int]:
        """Use the image batch dimension; E3A does not require GT extrinsics."""
        keys_to_log = self._get_scalar_log_keys(phase)
        images = batch["images"]
        if not torch.is_tensor(images) or images.ndim != 5:
            raise ValueError(
                "E3A training batches must contain images with shape [B,T,3,H,W]."
            )

        scalar_values: dict[str, float] = {}
        for key in keys_to_log:
            if key in batch:
                value = batch[key].item() if torch.is_tensor(batch[key]) else batch[key]
                scalar_values[key] = float(value)
        return scalar_values, int(images.shape[0])
