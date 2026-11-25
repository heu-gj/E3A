
# HOI4D Official Data Catalog and E3A Mapping

**Audit date:** 2026-11-25  
**Status:** Official source catalog: **PASS**; release-list reconciliation: **NEEDS_REVIEW**; local package crosswalk: **PARTIAL**; official 4D-seg generation: **BLOCKED**.  
**Purpose:** Define what HOI4D officially publishes, distinguish direct assets from derived labels and external dependencies, and map the published structure to the existing E3A local audit evidence. This document is not a claim that every supervision source is physically validated.

## 1. Source hierarchy and pinned references

Use the official project page for the published download entry points and data license. Use the dataset-instructions repository linked by that page for the data organization and format contract. Use the official challenge repositories only for task-specific evaluation conventions.

| Source | Pinned revision / evidence | Use |
|---|---|---|
| [HOI4D project page](https://hoi4d.github.io/) | Page last updated 2026-03-27 | Official download categories, project-level dataset description, data license |
| [HOI4D-Instructions](https://github.com/leolyliu/HOI4D-Instructions) | main `15478666c44af86740d537e89b72cd47efeb44f8` (2023-08-26) | Primary dataset instructions linked from the project page |
| [HOI4D_Script](https://github.com/hoi4d/HOI4D_Script) | main `788b43552176c77c01c3e98dd5f101fd6a94ee7a` (2023-02-18) | Earlier official-script repository; its test list differs from the linked instructions repository |
| [HOI4D_ActionSeg](https://github.com/hoi4d/HOI4D_ActionSeg) | main `ca779d86ad3dbf03275e0bc02bf4f9a5cf025884` | Action-segmentation challenge interface |
| [HOI4D_SemSeg](https://github.com/hoi4d/HOI4D_SemSeg) | main `66df3ff3f0465a9863781aace4d4f16bc204a43e` | Semantic-segmentation challenge interface |

The project page exposes six download entries: **RGB Video, Depth Video, CAD Models, Annotations, Camera Parameters, Hand Pose**. The linked OneDrive share pages did not expose their archive indexes through this audit. Therefore, package-level contents below are grounded in the official instructions' documented directory structure and in the already committed local audit—not in an independent enumeration of the remote ZIP members.

The dataset page states **CC BY-NC 4.0** for the dataset. The public instructions/script repositories have MIT code licenses. These are different licenses; do not infer that the dataset is commercially usable from the code license.

## 2. Official direct-download assets

### 2.1 RGB Video

Official sequence path:

~~~text
<sequence>/align_rgb/image.mp4
~~~

The official decoder uses FFmpeg, starts frame numbering at zero, and applies fps=15, writing five-digit zero-padded JPEG names (%05d.jpg). This is a decode convention, not proof of sensor capture timestamps or synchronization with every annotation stream.

### 2.2 Depth Video

Official sequence path:

~~~text
<sequence>/align_depth/depth_video.avi
~~~

The official decoder writes five-digit zero-padded PNG names (%05d.png). The released stream must be decoded without silently converting the raw depth values to ordinary RGB. The official README does not itself certify the physical unit of every decoded depth value.

### 2.3 CAD Models

The official instructions describe:

- Rigid object: one mesh at {object category}/{object id}.obj; the mesh defines the object's canonical frame.
- Articulated object: part meshes, a part hierarchy in result.json, joint annotations in mobility_v2.json, and per-part canonical meshes named {part name}_align.obj.
- Articulated joint metadata includes joint origin, direction, and rotation or translation limits.

CAD is a separate project-page download, not a file listed under each sequence's 3Dseg directory. A sequence's N* path component must not be assumed to be a CAD object ID without a verified join.

### 2.4 Annotations

The official instructions document the following sequence-level tree:

~~~text
<sequence>/
├── objpose/
│   └── *.json
├── action/
│   └── color.json
├── 2Dseg/
│   └── mask/
└── 3Dseg/
    ├── raw_pc.pcd
    ├── output.log
    └── label.pcd
~~~

The annotation package therefore covers several distinct supervision types. They must not be collapsed into one generic "3D labels" concept.

**Action segmentation — action/color.json**

- The official format description identifies video duration, event label, startTime, and endTime.
- The sequence path supplies category context (C*); the event string alone is not a globally unique action identity.
- The documented interval annotation does not itself identify a target entity, hand-object contact, or a persistent object track.

**2D Motion Segmentation — 2Dseg/mask/*.png**

- The official description says these masks annotate the human hands and objects related to the interaction.
- RGB palette colors are decoded with the official utils/color_map.py; semantic meaning is category-dependent and is defined by definitions/motion segmentation/label.csv.
- This is interaction-related motion segmentation, not an exhaustive scene-instance inventory. An unlabelled pixel/object is not automatically background, absent, or a negative training example.

**3D Static Scene Panoptic Segmentation — 3Dseg/**

- raw_pc.pcd and label.pcd describe the reconstructed static scene and its labels.
- definitions/3D segmentation/labels.xlsx contains the label definitions.
- output.log contains the per-frame camera pose/trajectory as consumed by the official Open3D example.
- Static-scene labels are not equivalent to per-frame dynamic object masks or persistent dynamic object tracks.

**Category-level Object Pose — objpose/*.json**

- The official example supports both dataList and objects containers.
- The example reads center, rotation, and dimensions; the README describes translation, rotation, and scale of a part.
- The raw record also includes fields such as frameId, filePath, isEffective, object id/label, and properties in the locally audited variant.
- A mathematically valid rotation matrix does not establish the physical coordinate frame, handedness, unit, CAD correspondence, or temporal identity.

### 2.5 Camera Parameters

Camera Parameters is a separate project-page download. In the local P1 audit, the extracted layout was:

~~~text
extracted/camera_params/camera_params/<camera_id>/intrin.npy
~~~

The sequence's 3Dseg/output.log is a separate source of per-frame camera poses. Do not conflate camera intrinsics with the per-frame trajectory, and do not substitute a generic Kinect2 calibration for the released camera intrinsics.

### 2.6 Hand Pose

Hand Pose is a separate project-page download. The official instructions describe one .pickle record per annotated hand frame with:

- poseCoeff: 3 global-rotation parameters plus 45 MANO pose parameters;
- beta: 10 MANO shape parameters;
- trans: translation in the camera frame;
- kps2D: 21 projected 2D keypoints.

The official example reconstructs 3D joints and vertices through manopth, divides the MANO output by 1000, and adds trans to obtain camera-frame coordinates in meters. This is a documented reconstruction recipe, not proof that E3A's FK, joint ordering, visibility, frame pairing, or camera projection has passed validation.

manopth source and the licensed MANO model assets are external dependencies. They are not the same thing as the HOI4D Hand Pose parameter package. The MANO model assets require their own access/usage permission.

## 3. Derived data: 4D Dynamic Scene Panoptic Segmentation

The official README says 4D dynamic panoptic labels are generated by prepare_4Dseg/prepare_4Dseg_dataset.py; it documents output under semantic_segmentation_label/*.txt, with semantic and instance labels in the final columns. This is a **derived output**, not a seventh direct-download entry on the project page and not part of the documented raw per-sequence tree.

The public generator is not currently safe to treat as a plug-and-play source of E3A GT. The linked instructions repository contains the following concrete source-level inconsistencies:

1. prepare_datalists.py: the datalist.append(...) statement is outside the inner for T in os.listdir(p6) loop. The label.txt check therefore does not gate the appended last T as intended, and the code does not enumerate every task directory correctly.
2. prepare_4Dseg_dataset.py: data_dir is parsed as a string, but the script iterates for seg2D_dir in data_dir, i.e. over characters, while trying to discover a refine_2Dseg directory.
3. The generator requests 3Dseg/label.txt and refine_2Dseg; the documented raw layout lists 3Dseg/label.pcd and 2Dseg/mask. The public instructions do not document the conversion/provenance that would bridge these names.
4. scene2frame.py reads unpadded 0.jpg / 0.png paths, while the official decoder writes zero-padded %05d.jpg / %05d.png.
5. scene2frame.py uses Open3D's hard-coded Kinect2ColorCameraDefault intrinsic instead of explicitly loading the separately published camera intrinsics.
6. The generator imports additional runtime dependencies, including Open3D, plyfile, pandas, SciPy, OpenCV, and NumPy. Dependency installation alone does not resolve the input-contract and calibration problems.

These are defects/assumptions in the public generation path, not evidence that the user's official download is incomplete. Do not re-download or re-extract data to compensate for undocumented derived-data prerequisites. Do not enable 4D-derived entity IDs as GT until the inputs, generator, camera model, output semantics, and real-sequence correspondence are independently validated.

## 4. Official category vocabulary

The release list contains 16 category codes:

| Code | Official category |
|---|---|
| C1 | Toy Car |
| C2 | Mug |
| C3 | Laptop |
| C4 | Storage Furniture |
| C5 | Bottle |
| C6 | Safe |
| C7 | Bowl |
| C8 | Bucket |
| C9 | Scissors |
| C11 | Pliers |
| C12 | Kettle |
| C13 | Knife |
| C14 | Trash Can |
| C17 | Lamp |
| C18 | Stapler |
| C20 | Chair |

The README's positional category mapping has blank entries for C10, C15, C16, and C19. The linked instructions' test list contains C10, C15, and C16 paths even though those codes are not in release.txt or the 16-row action/motion category tables. Do not invent category names for blank codes or silently include them in the released training vocabulary.

## 5. Release-list reconciliation — NEEDS_REVIEW

The official list files are not mutually consistent. Counts below are from exact line parsing of the pinned repository files; "unique" means unique full seven-component sequence paths.

| Source list | Lines | Unique paths | Duplicates |
|---|---:|---:|---:|
| Instructions release.txt | 2,971 | 2,971 | 0 |
| Instructions prepare_4Dseg/datalists/train_all.txt | 2,973 | 2,973 | 0 |
| Instructions testset.txt | 933 | 897 | 36 |
| Instructions prepare_4Dseg/datalists/test_all.txt | 933 | 897 | 36 |
| Older hoi4d/HOI4D_Script/testset.txt | 892 | 892 | 0 |

Verified set relationships:

- train_all.txt contains all 2,971 release.txt paths plus two additional paths: ZY20210800001/H1/C20/N32/S292/s04/T3 and ZY20210800003/H3/C3/N44/S284/s01/T2.
- Instructions release.txt and Instructions testset.txt share 31 unique paths.
- Instructions train_all.txt and test_all.txt share 32 unique paths.
- Instructions test_all.txt is byte-identical to Instructions testset.txt (same blob SHA).
- The older 892-path test list differs from the linked Instructions list by seven unique paths (one older-only and six Instructions-only).
- The ActionSeg and SemSeg challenge READMEs describe prediction arrays with 892 test samples. This benchmark shape is task-specific evidence; it does not by itself resolve the path-level list discrepancy.

Pinned list blob SHAs:

- Instructions release.txt: a528d66bfb2e757b9b8bf502113569a281f98114
- Instructions testset.txt / test_all.txt: fbf0fcdaf9b4d052961b3477611103ebceab0abf
- Instructions train_all.txt: 9a68a6d2a49fd6d0d291177b34f3eae105fb374f
- Older hoi4d/HOI4D_Script/testset.txt: afd84968c5591ba7f8afa9f4c4cb57d0798f53d6

**Policy for E3A:** Do not use train_all.txt or test_all.txt as a leak-free split. Do not union or deduplicate the official lists and assume the resulting set is a valid benchmark. Preserve the original list source and membership, resolve the task-specific evaluation protocol, then create E3A subject/sequence-level splits with explicit overlap checks.

## 6. Local package crosswalk from the committed P1 audit

The P1 report records the following extracted layout. These are local paths observed in the existing audit; they are not newly inferred paths.

| Official asset | Local extracted layout recorded by P1 | Status in current evidence |
|---|---|---|
| RGB Video | extracted/release/HOI4D_release/<sequence>/align_rgb/image.mp4 | Observed; P1 scanned 2,973 RGB sequence directories |
| Depth Video | extracted/depth/HOI4D_depth_video/<sequence>/align_depth/depth_video.avi | Observed; selected-sequence decode audit completed |
| Annotations | extracted/annotations/HOI4D_annotations/<sequence>/{2Dseg/mask,objpose,action,3Dseg} | Observed; one selected sequence received detailed physical/semantic audit |
| Hand Pose | extracted/hand_pose/Hand_pose/handpose_right_hand/<sequence>/<frame>.pickle and corresponding handpose_left_hand | Observed; PR #12 reconciles all 490,584 local pickle files against the available archive |
| Camera Parameters | extracted/camera_params/camera_params/<camera_id>/intrin.npy | Observed in the P1 audit for four camera IDs |
| CAD Models | No local CAD package path was recorded in the P1 sequence inventory | **NOT_MAPPED**; this is not evidence that the user's package is absent |
| Derived 4D labels | No generated dynamic 4D output was established for the P1 selected sequence | **NOT_ESTABLISHED**; do not confuse this with a missing direct-download package |

The P1 audit selected ZY20210800001/H1/C20/N30/S295/s04/T4 for detailed inspection. It recorded 300 RGB frames, 300 Depth frames, 300 motion masks, 300 object-pose JSON files, one action JSON, three static-3D files, 234 Right-hand pickles, and 233 Left-hand pickles. This is a selected-sequence audit, not a claim that every local sequence has identical coverage.

The P1 report records 2,973 local RGB sequence directories. This count numerically matches train_all.txt, but a count match is not path-set equivalence. The sanitized P1 evidence does not provide a complete path-by-path local inventory for comparison with every official list. Do not infer missing data from the 2,971-versus-2,973 count difference.

PR #12's hand-pose archive reconciliation found 497,277 archive entries and 490,584 hand-pose file streams; all 490,584 local pickles matched the available archive with no missing, extra, mismatched, CRC/read-error, or duplicate-conflict cases. The verdict remains: extraction equivalence **PASS**, official download completeness **NOT_VERIFIABLE** (no authoritative complete-download manifest). This result does not validate MANO FK or physical hand-pose GT.

## 7. What E3A may and may not supervise from HOI4D

| E3A task | Official source | Safe interpretation now |
|---|---|---|
| RGB appearance | RGB Video | Input stream after frame-index/FPS contract is audited |
| Depth / geometry | Depth Video + Camera Parameters + output.log + static 3Dseg | Geometry supervision only after units, intrinsics/extrinsics, timing, and alignment are validated |
| Entity masks | 2D Motion Segmentation; optionally derived 4D labels after validation | Partial interaction-related masks; not exhaustive scene instances |
| Object pose | objpose/*.json + matching CAD model | Candidate category-level pose; coordinate frame, scale, CAD join, and temporal semantics remain to be validated |
| Hand pose | MANO parameter pickles + external MANO assets | Reconstruct 3D with FK, then validate camera projection, joint order, units, frame association, and visibility |
| Contact | No direct contact label documented in the official format | Must remain invalid until source-backed contact targets or validated pseudo-labels exist |
| Future target entity | No explicit action-to-object identity field documented in the action interval schema | Must remain invalid until an event-specific target-to-observed-track bridge is verified |
| Time-to-event | Action interval start times | Can be derived after timestamp and observation-cutoff semantics are verified; this does not provide target identity |

Never infer a target object from the category code, N*, action name, object-pose id, or a provisional mask instance number alone. Preserve source namespaces separately.

## 8. Immediate consequences for implementation

1. Freeze the official source revisions and list-file SHAs before creating any E3A split.
2. Keep release.txt, testset.txt, train_all.txt, and challenge benchmark sample counts as separate source facts; do not silently substitute one for another.
3. Finish the CAD package local-path mapping as a distinct inventory task. Until then, label it NOT_MAPPED, not MISSING.
4. Keep 4D Dynamic Scene Panoptic Segmentation disabled as a training source until the official generator's path, input, calibration, and output contracts are repaired and validated.
5. Keep 2D motion-mask unknown pixels and unannotated frames invalid; do not convert them to background or no-object labels.
6. Keep MANO FK, contact, and target-entity validity gated by their own physical/semantic validation.
7. Do not re-download, re-extract, or repeat blanket SHA256 scans without a specific discrepancy that requires them.

## 9. Audit verdict

- **Official direct-download categories:** PASS (six categories listed by the project page).
- **Official raw sequence organization and annotation formats:** PASS (documented in the linked instructions).
- **Derived 4D segmentation as an independent direct-download asset:** NOT_SUPPORTED by the published download list; it is documented as script-generated.
- **4Dseg generator runnable/physically validated as published:** BLOCKED.
- **Official train/test list semantics and path-level consistency:** NEEDS_REVIEW.
- **Local package crosswalk:** PARTIAL; grounded in the existing P1 report.
- **Local hand-pose extraction/archive equivalence:** PASS within the available archive scope; official full-download completeness remains NOT_VERIFIABLE.
- **E3A physical target/contact supervision readiness:** NOT_READY.

This catalog defines the data boundary. The next work must address the concrete list/split and adapter-contract questions without treating official omissions or ambiguous derived-data prerequisites as local download failures.
