# HOI4D hand-pose coverage audit

Run from the repository's approved environment:

```bash
python scripts/audit_hoi4d_handpose_coverage.py \
  --root "$HOI4D_ROOT" \
  --output-dir "$AUDIT_OUTPUT"
python -m pytest -q tests/test_hoi4d_handpose_coverage.py
```

The output directory must be outside the dataset and should remain outside Git.
An existing inventory database is not overwritten. The tool reads every local
hand pickle with the restricted production parser, records per-file schema and
SHA256 in JSONL/SQLite, enumerates local RGB/annotation sequences independently,
and probes video metadata without decoding the videos. It never extracts archives,
downloads data, reconstructs MANO joints, or produces training labels.

Sequence identity is the full seven-component release path. No separate take UID
is supplied by this layout; the operational take count equals the sequence count,
and no cross-camera take correspondence is inferred. Frame numbers are filename
indices checked against video bounds, not verified sensor synchronization.

The universe covers all *locally extracted* sources. It cannot establish the
official release's complete sequence population without an official manifest.
Absent hand files remain unknown/unannotated, not corruption. Numeric field
extrema are diagnostics; no invented physical thresholds are used. A successful
MANO parser result does not validate pose conventions, units, joints or contact GT.

ZIP/TAR members, when available, are streamed without extraction and compared by
SHA256 to extracted hand files. ZIP streams validate CRC on reading. Extraction
equivalence applies only to available local archives; download completeness stays
NOT_VERIFIABLE without authoritative checksums or a completeness manifest.

Reports, per-file inventories and source checksums are local research evidence.
Do not commit raw pickles, masks, video, full real-data reports or sensitive paths.
