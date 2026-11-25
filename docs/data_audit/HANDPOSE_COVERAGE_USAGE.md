# HOI4D hand-pose coverage audit

Run from the repository's approved environment:

```bash
python scripts/audit_hoi4d_handpose_coverage.py \
  --root "$HOI4D_ROOT" \
  --archive "$HANDPOSE_ARCHIVE" \
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

For archive-only reconciliation (without repeating MANO parameter parsing):

```bash
python scripts/audit_hoi4d_handpose_archive.py \
  --root "$HOI4D_ROOT" --archive "$HANDPOSE_ARCHIVE" \
  --output-dir "$AUDIT_OUTPUT"
```

`--archive` is repeatable and supports files outside the dataset root. Results
separate `ARCHIVE_READABILITY`, `ARCHIVE_CRC_INTEGRITY`,
`EXTRACTION_EQUIVALENCE` and `DOWNLOAD_COMPLETENESS`. Empty/no-hand archives,
extra local files, duplicate/conflicting members, unsafe paths and read errors
cannot yield equivalence PASS. Content disagreement is NEEDS_REVIEW, not proof
that either source should be overwritten. Missing extracted files block
equivalence; the tool never extracts or repairs them.

Windows separators are normalized only when the exact side/seven-component
sequence/filename hierarchy is unambiguous. All member occurrences remain in
the disk-backed inventory, including rejected paths and duplicate members.
Full ZIP bytes are hashed before/after; every non-directory member is streamed
to EOF for CRC, and local hand files are independently hashed twice. SQLite and
per-file CSV outputs contain sensitive relative identifiers and remain local.

Reports, per-file inventories and source checksums are local research evidence.
Do not commit raw pickles, masks, video, full real-data reports or sensitive paths.
