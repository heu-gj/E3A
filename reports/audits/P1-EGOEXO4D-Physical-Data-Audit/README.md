# P1-EGOEXO4D-Physical-Data-Audit

Status: **PARTIAL**. [Report](report.md), [structured results](report.json),
[manifest](manifest.json), [supervision gates](evidence/supervision_readiness.json).

The local metadata explicitly identifies V2. Real Relations annotations and
downscaled Aria/Exo videos were audited. EgoPose, camera poses, Raw Keystep and MPS
files are missing from the checked local root. Missing measurements are
BLOCKED/NOT_RUN, not synthetic substitutes or zero reprojection errors.

Take: `51fc36b3-e769-4617-b087-3826b280cad3`, `georgiatech_cooking_01_01_2`.
No model, production adapter, loss or trainer changes. No training or dataset writes.
Target/contact validity remains false.

All source dataset paths are relative to the configured dataset root. Code paths
are repository-relative; evidence references are archive-relative. Only metadata,
statistics, sanitized logs and code are published. Raw selected annotation and
unapproved images remain Git-ignored locally; their hashes/size are in manifest.

Run `sha256sum -c SHA256SUMS.txt` from this directory. The checksum file includes
manifest.json but excludes itself. Manifest hashes exclude itself and the checksum
file to avoid recursion. Its current_commit is the parent observed when sealing;
the final delivered commit cannot embed its own hash and is shown by PR history.

Evidence includes full metadata candidate inventory, selected track coverage,
70 real mask-decode observations, bounded clip targets, sampled video PTS,
adapter review, and explicit missing-data records. Tests are distinct from live
data measurements; see logs/tests.json and logs/commands.txt.
