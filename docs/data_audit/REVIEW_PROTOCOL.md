# Audit Artifact Archival Protocol v1.0

Execution results belong only in `reports/audits/<task_id>/`. Reusable code belongs
in `scripts/`, tests in `tests/`; this document is the shared review protocol.

## Evidence and status

- PASS: the stated check ran and has an explicit evidence file and field.
- PARTIAL: some checks ran; record their limits and remaining gaps.
- BLOCKED: a required dependency, annotation or authority is missing.
- NEEDS_REVIEW: an observation exists but its physical or semantic meaning is unverified.
- NOT_RUN: the check did not execute; never imply it passed.

Separate parser/unit tests from real-data validation. Matching filenames do not
prove synchronization. SO(3) matrices constructed from Euler parameters do not
validate physical coordinate direction, units or object identity. Missing labels
are not negative examples. Do not invent GT or silently normalize ambiguous labels.

## Archive layout

Each task has README.md, report.md, report.json, manifest.json, evidence/,
visualizations/, logs/ and SHA256SUMS.txt. Keep one canonical final report.
Before consolidating duplicates, compare contents and record hashes. Use `git mv`
for tracked or intentionally staged sanitized text. Initially untracked restricted
binary artifacts can be relocated locally without staging; record that exception.
Do not delete source evidence or modify original datasets.

Dataset references are relative to the dataset root. Repository, archive and
external-source references use explicitly named namespaces; never confuse them
with dataset references. Remove local absolute paths and secrets from published
reports, code defaults and logs. External URLs are allowed when they are sources,
not local filesystem paths.

## Publication boundary

Publish metadata/statistics, sanitized text/code/logs and only explicitly cleared
derived visualizations. When visualization permission is uncertain, withhold it.
Never stage original videos, masks, depth arrays, annotation archives, complete
source annotation copies, model weights/checkpoints or secrets. Preserve withheld
material in a Git-ignored local-only area and record its relative path, size,
checksum and reason in the public manifest. A clone must clearly distinguish
withheld local evidence from missing public files.

## Provenance and checksums

Record repository, source branch/base commit, current archival parent commit,
UTC execution time, dataset version (UNKNOWN if not confirmed), sequence, source
inventory, published artifact inventory and SHA256. SHA256SUMS.txt covers all
published payload files including manifest.json, excluding itself. The manifest
contains payload hashes excluding itself and SHA256SUMS.txt to avoid a hash cycle.
Withheld hashes describe local files and are not required by public checksum checks.

A commit cannot contain its own final hash. `audit_commit` must therefore identify
the immutable commit with the audit payload; a later metadata-only commit may pin
that hash. Define `current_commit` as the parent observed when sealing metadata,
not a claim that the file embeds the hash of its own future commit. Report the
final pushed HEAD separately in the handoff and PR.

## Git and review

Inspect status, branch, commit and existing PRs first. Preserve unrelated user
changes. Reuse an existing task PR; otherwise create the requested task branch.
Run relevant tests, `git diff --check`, `git diff --cached --check`, staged-file
allowlist, path/secret checks and size checks before committing. Push and open a
PR against the designated base; never merge during archival. Stop for independent
review after reporting branch, HEAD, PR, archive, tests and unresolved issues.
