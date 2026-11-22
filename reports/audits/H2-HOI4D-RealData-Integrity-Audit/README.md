# H2 HOI4D real-data integrity audit

Start with [report.md](report.md), [report.json](report.json),
[manifest.json](manifest.json) and [summary metrics](evidence/summary_metrics.json).
The sole audited sequence is `ZY20210800001/H1/C20/N30/S295/s04/T4`.

This compares actual H1 Dataset outputs with a separate sequentially decoded source
reference, over all300 frames. It does not establish physical units, calibration,
3D hand joints, persistent entity identity, contact or action-target labels.
The final status is governed by the real evidence, with digital/physical scopes
reported separately. Source data is never modified; no model or training is run.

Code: `scripts/audit_hoi4d_realdata.py`; tests: `tests/test_hoi4d_realdata_audit.py`.
Use repository `.venv` only. pytest installation was explicitly approved; no other
new packages were installed. Prior Conda environments are unchanged.

Data references are dataset-root-relative, artifact references archive-relative,
and code paths repository-relative. No host absolute paths or credentials are
published. Raw reference arrays and unapproved images stay in Git-ignored local_only
folders, with relative path/size/SHA256 in manifest. A public clone will not contain
these files. Unknown pixels and missing labels are not converted to valid GT.

Run `sha256sum -c SHA256SUMS.txt` from this directory. The manifest excludes its own
hash and SHA256SUMS to avoid recursion; SHA256SUMS includes manifest but excludes
itself. Embedded current_commit denotes parent HEAD observed at sealing; the final
delivered commit appears in PR history. This audit stops at H2; never merge or start H3.
