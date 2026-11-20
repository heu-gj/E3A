## Local Ego-Exo4D physical supervision audit

Status: **PARTIAL**. Base: `feat/e3a-foundation` at
`bade1b8d68205335ce039490ded919965216fe70`, the verified squash merge of PR1.
No model, production adapter, loss or trainer changes; no training or source-data writes.

The local metadata explicitly says V2. The checked root contains real Relations
JSON and11775 downscaled videos, but lacks EgoPose hand/camera annotations, Raw
Keystep, keystep_benchmark and actual MPS assets. Metadata-advertised VRS/trajectory
paths do not exist. Missing hand/geometry/event measurements are BLOCKED/NOT_RUN,
with NULL reprojection errors; synthetic parser tests are not substitutes.

Selected take: `51fc36b3-e769-4617-b087-3826b280cad3` /
`georgiatech_cooking_01_01_2`, ranked from695 available ego/exo multi-object candidates.
It has16 non-hand tracks. Official Ego4D SDK1.7.3 decoded70 real mask records;
all agree with the E3A decoder and source dimensions. Coverage checks found no
mask outside annotated_frames in25 track/camera records. A real two-track clip
preserves explicit absence separately from unannotated frames.

Aria and cam01 each report31048 frames at30FPS. Five temporal positions per camera
and exact nearby annotated frames were read separately; local overlays support
coarse correspondence after scaling, not calibrated synchronization. Original
mask dimensions differ from448x448/796x448 videos. The production resolver fails
because metadata original-resolution paths are absent and only downscaled paths
exist. aria01 and aria04 stream aliases were checked against actual metadata.

Potential first-record-only EgoPose data loss is documented with a constructed
adversarial test, not misreported as an observed local release bug.137 train
Relations UIDs have no counterpart in local takes metadata; cause remains UNKNOWN.

Only conditional Entity mask supervision is supported. Hand, Geometry, Interaction,
Target, Action and Time remain blocked by missing source/verified bridges. Object
names are not event targets; target_entity_valid and interaction_valid stay false.

Canonical archive: `reports/audits/P1-EGOEXO4D-Physical-Data-Audit/`.
31 tests passed (23 existing,8 helper/artifact regressions), separate from real
measurements. SHA256 and publication-boundary checks pass. Images and raw selected
annotations stay local-only; published files contain metadata/statistics and logs.

Independent review requested. **Do not merge; do not proceed to P2.**
