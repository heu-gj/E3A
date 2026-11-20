# ADR 0001: Upstream-first implementation

- Status: Accepted
- Date: 2026-11-20

## Decision

E3A will track official research implementations as pinned Git submodules and keep E3A-specific code in src/e3a. It will not copy large upstream model implementations into the E3A source tree.

## Rationale

- The official implementation is the strongest source of truth for tensor shapes, preprocessing, checkpoint keys, model configuration, and inference lifecycle.
- Submodule pins make experiments reproducible and make upstream updates reviewable.
- Thin adapters reduce accidental API drift.
- Minimal patches are easier to audit than forks with broad edits.

## Rules

1. Never track a floating upstream branch as the experiment source of truth; pin a commit.
2. Do not vendor pretrained checkpoints or datasets.
3. Do not use strict=False to hide unexplained checkpoint mismatches.
4. Every upstream patch must include the exact upstream commit, patch rationale, and parity tests.
5. Preserve the default upstream forward behavior when adding optional E3A features.
6. Check each upstream license before copying, modifying, redistributing, or using weights.
7. If an upstream project has no clear license, use it as a method reference only until permission is confirmed.

## Consequences

E3A's first milestones prioritize reproducible dependency setup and parity tests over implementing all prediction heads at once. A trainable model is not claimed until dataset contracts, tensor contracts, and loss semantics have been validated.
