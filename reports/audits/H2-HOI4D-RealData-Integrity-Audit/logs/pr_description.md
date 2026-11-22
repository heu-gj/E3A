## H2 HOI4D real-data integrity and visualization audit

Base commit: `155820d65c2f8e47101d5aa204d0fcea79cb6776`.
All execution uses repository `.venv` (Python3.10.19, torch2.6.0 CPU, NumPy1.26.4).
Only pytest/light dependencies were installed after explicit user approval.

**Digital integrity: PASS. Overall physical supervision audit: PARTIAL.**
The real sequence `ZY20210800001/H1/C20/N30/S295/s04/T4` was compared against
independent sequential source reads over300 unique frames/304 observations.
Seed220 shuffles all38 Dataset clips. Full raw seek hashes, RGB area resize and
normalized tensors, uint16-depth values retained in float32, mask/validity,
raw objpose fields, MANO parameters/2D scaling and K scaling agree exactly.

The H1 smoke rerun succeeds:300/300 frames,38 batches,467 valid hand frames,
14 action events,34 valid Time clips,0 Action clips without a vocabulary.
Frame220's37 undefined palette4 pixels invalidate16 downsampled pixels while
known Chair foreground survives. Invalid zeros are placeholders, not negative GT.
Pose filename indices remain separate from mixed raw frameIds.304 pose reads
include four duplicated source frames; there are only300 source JSON files.

Thirty-two local visualization positions include0/75/150/220/224/299 plus
actual action-boundary neighbors. All images/reference arrays are Git-ignored;
the public manifest lists local relative paths, sizes and SHA256 values.

37 tests pass, including new metric/unknown/index/depth/missing-hand/future-boundary
helpers and actual-evidence regressions. Full source measurements are separately
recorded; synthetic tests are not physical validation. Checksums, staged-publication
boundaries and whitespace checks pass.

No production Dataset/decoder, model, heads, loss or trainer changes. Physical
units, sensor sync, RGB-depth registration, objpose direction, MANO3D reconstruction,
persistent physical identity, contact and future-target relationships remain
unverified. No valid Target/Interaction GT is invented. Downstream loss consumption
of pixel_valid is NOT_RUN, not assumed from tensor presence.

Canonical archive: `reports/audits/H2-HOI4D-RealData-Integrity-Audit/`.
Please review independently. **Do not merge; do not proceed to H3.**
