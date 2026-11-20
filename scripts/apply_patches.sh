#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VGGT_DIR="$ROOT/third_party/vggt-omega"
PATCH_FILE="$ROOT/patches/vggt-omega/0001-extra-tokens-and-dense-features.patch"

if [[ ! -d "$VGGT_DIR/.git" ]]; then
  echo "VGGT-Ω submodule is missing. Run git submodule update --init third_party/vggt-omega." >&2
  exit 1
fi

if git -C "$VGGT_DIR" diff --quiet -- vggt_omega/models/aggregator.py vggt_omega/models/heads/dense_head.py; then
  git -C "$VGGT_DIR" apply --check "$PATCH_FILE"
  git -C "$VGGT_DIR" apply "$PATCH_FILE"
  echo "Applied the pinned E3A VGGT-Ω compatibility patch."
else
  echo "VGGT-Ω source has local modifications. Refusing to apply the patch twice or overwrite edits." >&2
  exit 1
fi
