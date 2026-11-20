#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git submodule update --init --recursive
bash scripts/apply_patches.sh

python -m pip install -e ".[dev]"
python -m pip install -e 'third_party/vggt-omega[train]'

if [[ "${E3A_INSTALL_SAM3:-0}" == "1" ]]; then
  echo "Installing optional SAM 3.1 package. Verify the official Python/PyTorch/CUDA requirements first."
  python -m pip install -e third_party/sam3
fi

echo "E3A base environment installed."
echo "SAM 3.1 is optional; set E3A_INSTALL_SAM3=1 only in a compatible environment."
