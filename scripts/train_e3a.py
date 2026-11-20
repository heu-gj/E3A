"""Launch E3A training with VGGT-Ω's Hydra/DDP/FSDP infrastructure."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from e3a.engine.trainer import E3ATrainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "training"


def load_config(config_name: str = "e3a", overrides: list[str] | None = None):
    config_file = CONFIG_DIR / f"{config_name}.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"Training config {config_file} is not present yet. "
            "Create it only after the E3A dataset and label contracts are audited."
        )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name=config_name, overrides=list(overrides or []))
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)
    config.pop("exp_name", None)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train E3A")
    parser.add_argument("--config", default="e3a")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    E3ATrainer(**load_config(args.config, args.overrides)).run()


if __name__ == "__main__":
    main()
