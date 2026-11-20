#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VGGT_DIR="$ROOT/third_party/vggt-omega"

if [[ ! -e "$VGGT_DIR/.git" ]]; then
  echo "VGGT-Ω submodule is missing. Run git submodule update --init third_party/vggt-omega." >&2
  exit 1
fi

apply_one_patch() {
  local patch_file="$1"
  shift

  if git -C "$VGGT_DIR" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "Already applied: $(basename "$patch_file")"
    return
  fi

  if ! git -C "$VGGT_DIR" diff --quiet -- "$@"; then
    echo "VGGT-Ω has unrelated local modifications in files touched by $(basename "$patch_file")." >&2
    echo "Refusing to overwrite edits." >&2
    exit 1
  fi

  git -C "$VGGT_DIR" apply --check "$patch_file"
  git -C "$VGGT_DIR" apply "$patch_file"
  echo "Applied: $(basename "$patch_file")"
}

apply_one_patch "$ROOT/patches/vggt-omega/0001-extra-tokens-and-dense-features.patch" vggt_omega/models/aggregator.py vggt_omega/models/heads/dense_head.py
apply_one_patch "$ROOT/patches/vggt-omega/0002-official-forward-feature-api.patch" vggt_omega/models/vggt_omega.py
