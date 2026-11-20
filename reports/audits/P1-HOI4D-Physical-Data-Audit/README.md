# P1-HOI4D-Physical-Data-Audit

Final status: **PARTIAL**. Start with [report.md](report.md), [report.json](report.json)
and [manifest.json](manifest.json). Shared policy: [REVIEW_PROTOCOL.md](../../../docs/data_audit/REVIEW_PROTOCOL.md).

Identity & Geometry follow-up retracts the original constant-frameId-offset claim.
See report section14 and evidence/{objpose_frameid_audit,identity_mapping,
dynamic_4d_pipeline_audit,frame220_palette_audit,multi_object_target_audit}.json
plus objpose_frameid_audit.csv. The follow-up reread actual source files and used
HOI4D_Script commit `788b43552176c77c01c3e98dd5f101fd6a94ee7a`.
Original28 tests/logs remain; follow-up regression/helper checks have separate logs.

Published evidence is metadata/statistics only. All dataset paths are relative to
the dataset root; repository and archive references have separate namespaces.
Original videos, masks, depth arrays, full annotation copies, checkpoints, secrets
and unapproved derived images are not published. Local-only material is retained
and described by hashes in the manifest, but will not exist in a public clone.
See [visualization policy](visualizations/README.md).

The archive does not claim verified physical coordinates, persistent entity IDs,
contact labels, action targets or recovered MANO joints. Every PASS is scoped to
the evidence named in report.json. Parser/collation tests are synthetic; actual
dataset observations are recorded separately. Initial archival was not a rerun;
the subsequent follow-up is a new real-data audit, explicitly described in section14.

`base_commit` is the original audited source. `audit_commit` pins the payload
commit. `manifest.current_commit` is the HEAD observed when sealing metadata;
the final metadata commit cannot contain its own hash and is identified by the PR.

`SHA256SUMS.txt` covers every published archive file except itself, including
manifest.json. The manifest hashes published payloads except itself/checksum file
to avoid recursion. Run `sha256sum -c SHA256SUMS.txt` from this directory.

Files: README.md, report.md/json, manifest.json, SHA256SUMS.txt;
evidence/{observations,inventory,candidates,mask_statistics,objpose_statistics,
mano_right_statistics,mano_left_statistics}.json, evidence/cross_modal_frames.csv;
logs/{archival_provenance,tests}.json, logs/pytest.txt; visualizations/README.md;
.gitignore. Code remains in scripts/ and tests/.
