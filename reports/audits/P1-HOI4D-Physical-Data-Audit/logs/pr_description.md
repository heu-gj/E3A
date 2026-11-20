## HOI4D identity and geometry bridge audit

This PR archives P1 results and the follow-up corrections. Overall status: **PARTIAL**.
It targets `feat/e3a-foundation`; do not merge before independent review.

The follow-up retracts the original constant-offset frameId claim. All300 source
JSON files were reread:269 frameIds equal filenames, while31 nonempty-filePath
records use frameIds1..31. Field meaning and annotation timestamps remain UNKNOWN.
The updated CSV separates source indices, raw frameIds and candidate video PTS.

HOI4D code is pinned to `788b43552176c77c01c3e98dd5f101fd6a94ee7a`.
For the Chair sequence,31 available source boxes have bbox/mask IoU0.4999–0.6562;
all300 projected center hypotheses land in the Chair mask. These checks support
coarse correspondence, not exact synchronization or physical unit certification.

Open3D0.19 reads all300 log matrices as inverse(raw). Five-frame real static
reprojection diagnostics support that direction. The official4D generator was
NOT_RUN: label.txt/refine_2Dseg/plyfile are missing, and the pinned wrapper has
input/path inconsistencies. The static-max foreground offset rule was verified,
but conditional IDs are not presented as generated GT.

Pinned pixel2category was executed on real frame220 and skips its37 palette4
pixels. The proposed ignore=-1 policy is audit-only; production decoder is unchanged.

The selected C13/T4 sequence is `ZY20210800001/H1/C13/N12/S179/s03/T4`.
Apple has an independent palette3 mask in300/300 frames, while all300 objpose
objects are id2/knife. During the cut interval, imagery supports Knife as tool
and Apple as patient. No source event identifies a target instance, so Target
Entity supervision remains invalid; neither N* nor action names become target GT.

Validation: **37 tests pass**, retaining the original28 and adding9 regression/
helper checks. Synthetic and artifact regression tests are explicitly separate
from the real-data evidence. SHA256, publication-boundary and whitespace checks pass.
The canonical report and all public statistics are under
`reports/audits/P1-HOI4D-Physical-Data-Audit/`. Images, raw depth and full annotations
remain local-only; model/heads/loss/trainer and production adapters are unchanged.
