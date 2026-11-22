# Training framework reuse

E3A reuses the VGGT-Ω training implementation in third_party/vggt-omega/training:

- Trainer: DDP/FSDP setup, gradient accumulation, AMP/scaling, optimizer/scheduler construction, logging, checkpoint/resume, and the training loop.
- train_utils: distributed setup, checkpoint helpers, optimizer construction, gradient clipping, and device transfer.
- Hydra: component construction from E3A-owned configs.

E3A does not fork the training loop. E3ATrainer overrides only task-specific assumptions:

1. Pass batch timestamps to E3AModel along with RGB frames.
2. Derive batch size from the RGB tensor instead of requiring batch["extrinsics"].
3. Wrap predicted pose_enc in pose_enc_list when the official camera loss is retained.

## Model and objective

Instantiate e3a.models.e3a_model.E3AModel as the model and e3a.engine.losses.E3AMultiTaskLoss as the loss. E3AModel performs one shared VGGT-Ω forward and returns masks, 3D entity states, hand pose, pairwise interactions, target-entity logits, action logits, and time-to-event distribution parameters.

E3AMultiTaskLoss returns the single loss_objective required by the upstream Trainer. It internally calls VidEoMTMaskCriterion.forward_with_matches once; the returned clip-local GT-track-to-query-slot map is reused to supervise interaction pairs and target-entity selection. There is one optimizer and one joint backward pass.

## Canonical batch contract

Use e3a.data.collate.collate_e3a_batch to collate per-sample dictionaries. Every sample contains:

- images: [T,3,H,W] RGB float tensor
- timestamps: [T] seconds, monotonically non-decreasing
- entity_targets: labels [N], ids [N,T], masks [N,T,H,W], optional mask_valid [N,T], optional pixel_valid [T,H,W]
- optional hand_targets: keypoints_camera [T,2,21,3], valid [T,2,21], optional visibility and visibility_valid
- optional interaction_targets: contact [T,2,N], valid [T,2,N]
- optional anticipation_targets: target_entity_track_id, target_entity_valid, action_label, action_valid, time_to_event, time_valid

The collator returns batch-first tensors:
- images [B,T,3,H,W], timestamps [B,T]
- entity_targets labels [B,Nmax], ids [B,Nmax,T], masks [B,Nmax,T,H,W], mask_valid [B,Nmax,T], pixel_valid [B,T,H,W], track_counts [B]. If pixel_valid is absent, the adapter defaults it to all true.
- entity_supervision_valid [B]: true only when the clip has at least one entity track with valid entity supervision. A false clip is excluded from entity classification and mask losses; it is not interpreted as an all-background/no-object clip.
- hand_targets keypoints_camera [B,T,2,21,3], valid [B,T,2,21]
- interaction_targets contact/valid [B,T,2,Nmax]
- anticipation_targets one scalar per clip

Missing task annotations are represented by zero placeholders and false validity masks. They are not treated as negative examples. Per-track mask_valid is also padded false. The clip-level entity_supervision_valid flag suppresses all entity losses for empty/unannotated clips, including no-object classification, while other task losses remain independently validity-masked. The VidEoMT adapter retains the pinned clip-consistent matching policy, but when pixel_valid contains unknown pixels it samples Hungarian matching costs and BCE/Dice mask-loss points only from valid pixel centers. mask_valid filters unannotated track-frame pairs; pixel_valid filters unknown spatial locations. Unknown pixels therefore cannot contribute to mask matching costs or mask-loss gradients.

## Source annotation adapters

### HOI4D

- decode_hoi4d_motion_mask accepts an RGB palette image (not OpenCV BGR) and a category code such as C8. Its palette matches the official utils/color_map.py; category semantics are cross-checked against definitions/motion segmentation/label.csv. The 2D label.csv does not itself define persistent instance IDs, so part-to-instance grouping borrowed from prepare_4Dseg/utils.py remains provisional until audited on real 2D clips.
- build_hoi4d_entity_clip_targets groups non-hand semantic parts sharing the official instance ID into one object track. Semantic class 47 (hands) is excluded because E3A has a separate hand-pose head.
- A track/frame is valid only when that instance has visible pixels in the supplied motion mask. A missing image or an instance with no pixels is unannotated (mask_valid=False), not an explicit negative: HOI4D's motion mask does not distinguish full occlusion/out-of-view from true absence.
- parse_hoi4d_action_segments supports the documented single-event layout, an events-list layout, and a compatibility-only legacy markResult.marks layout. The official dataset instructions document the single-event duration/event/startTime/endTime form; the other variants must be confirmed against the locally downloaded release before use. Pass one of the 16 released 2D category codes when parsing so repeated event names from different object categories remain distinct. It keeps interval times in seconds and rejects intervals outside the declared duration. Confirm the selected layout and duration against the locally downloaded release before building clips.
- build_hoi4d_action_vocabulary must receive training-split segments only. build_hoi4d_anticipation_target selects the first event starting strictly after the observation cutoff. Unknown action labels disable action loss but preserve time loss; target entity remains invalid because the action interval does not identify a matched object.

### Ego-Exo4D

- parse_egoexo_hand_pose_annotations reads annotation3D world-frame joints and applies the per-frame camera_extrinsics as p_camera = R p_world + t. It supports both the documented frame-first JSON layout and the official baseline's camera-first layout.
- Manual EgoPose annotations use the official baseline's minimum three-view filter (min_views_for_3d=3); automatic annotations bypass that filter (min_views_for_3d=None). Select this from the annotation release/path, not by guessing from missing joints.
- Hand order is right then left. Within each hand the 21 joints are wrist, thumb(1..4), index(1..4), middle(1..4), ring(1..4), pinky(1..4), matching the official baseline reader.
- source_to_model_units defaults to 1.0 and preserves source units. The source annotation docs do not establish a physical unit here; set this scale only after auditing the downloaded release and VGGT depth units. Manual annotations can be filtered with min_views_for_3d=3.
- build_egoexo_relation_clip_targets uses annotated_frames as the coverage mask. A frame listed in annotated_frames without an encoded mask is explicit absence; a frame outside that list is unannotated. It excludes exact hand-track names (left/right hand with optional numeric suffix) by default, not unrelated object names that merely contain the word "hand".
- Relation encodedMask decoding uses the official Ego4D SDK decoder and pycocotools. Install the optional dependencies with pip install -e ".[egoexo]".
- resolve_egoexo_video_path follows the official Ego4D notebook's frame_aligned_videos/root_dir/relative_path convention, including the aria01_214-1 to rgb stream alias.
- parse_egoexo_keystep_segments reads the raw take-level start_time/end_time/step_unique_id schema. build_egoexo_keystep_vocabulary builds a deterministic vocabulary from the training split only; keep its source IDs dataset-prefixed when combining with HOI4D.
- build_egoexo_anticipation_target selects the first segment with start_time strictly after the observation cutoff. It creates action/time labels from future annotations while leaving target-entity supervision invalid because the keystep source does not identify the interacted object. Unknown action IDs disable only action loss, not time-to-event loss.

build_dataset_prefixed_action_vocabulary combines source labels without merging same-named classes across datasets. Use keys such as egoexo4d::step_unique_id=843 and hoi4d::pour. The Ego-Exo4D and HOI4D anticipation builders accept these prefixed keys; their dataset-only vocabulary helpers remain available for isolated experiments.

These are annotation-level adapters, not yet complete end-to-end Dataset/DataLoader implementations. Video frame decoding, release-version indexing, cross-source coordinate/unit auditing, building the actual vocabulary from audited training annotations, event-window sampling, and source-video-level split generation remain gated.

**Empty-clip safety:** the current VidEoMT adapter treats a clip with zero GT tracks as an all-no-object example. Do not pass a zero-track window merely because masks are missing, objects are out of view, or no relation track is visible in the sampled window. Until a clip-level entity_supervision_valid field is implemented, drop such windows unless the source explicitly establishes a fully annotated empty scene. HOI4D has no explicit per-track absence/occlusion flag in its motion masks; its adapter therefore marks an instance/frame valid only when pixels are visible. Ego-Exo4D relation annotated_frames can establish explicit absence for a listed object/frame.

## Current supervision status and training gate

The current source adapters can supervise HOI4D/Ego-Exo4D entity masks, EgoPose hand joints, and future action/time labels. They deliberately set target_entity_valid=False because neither source adapter yet establishes which observed entity is the target of the next action. Neither adapter currently emits interaction_targets with source-backed hand-object contact labels.

Consequently, on the current dataset adapters:
- loss_target_entity is zero. PersistentEntityPool, AnticipationHead's target-entity projection/no-target branch, and its future query projection receive no direct gradient.
- loss_interaction is zero. InteractionHead's contact_head receives no direct gradient. The pair encoder can still receive indirect gradients through action/time prediction.
- action/time prediction, entity-mask supervision, and available EgoPose hand-pose supervision can train independently.

Do not describe the current training setup as fully entity-grounded anticipation. To unlock that claim, add audited target-track mappings and contact labels (or explicitly validated pseudo-labels), then verify non-zero gradients and held-out metrics for the corresponding heads. Do not manufacture target labels by assuming the sequence's primary object is always the future action target: HOI4D tasks can involve secondary objects, and Ego-Exo4D Keystep labels do not identify an object track.

## Entity matching contract

The entity target contract remains:

- labels [N]: int64, zero for the single foreground class "entity"
- ids [N,T]: int64, stable non-negative clip-local track ID while visible, -1 for frames without a visible object
- masks [N,T,H,W]: binary/soft masks, zero wherever ids == -1
- mask_valid [N,T]: bool, true when the frame/track mask annotation is covered; false for unannotated frames. A valid frame with ID -1 is an explicitly annotated absence; an invalid frame is ignored by mask loss
- each GT track row has one stable ID throughout the clip and a non-empty mask on visible frames

VidEoMT's consistent matcher assigns each GT track to one query slot across the clip. That assignment is used for both mask loss and downstream target/interaction losses. The current target-entity head only supervises a target that is visible and matched in the observed prefix. An absent/unseen future target must be marked invalid until an explicit unseen-target class is designed and audited.

## Dataset-specific supervision

HOI4D and Ego-Exo4D retain their own raw taxonomies. Build a dataset-prefixed union action vocabulary rather than assuming similarly named classes are equivalent. The data adapters must provide explicit masks for annotation availability, occlusion, and missing frames. In particular, Ego-Exo4D annotated_frames records annotation coverage, not necessarily object visibility; only frames absent from that list are unlabeled, while frames in the list but without an encoded mask are explicit annotated absence.

HOI4D motion segmentation is palette-coded and must be decoded using the official category-to-semantic and category-to-instance maps. Its MANO hand annotations need to be converted to the canonical 21-joint order and camera-frame meters. Dataset-specific coordinate conversion must be tested before enabling the corresponding loss.

## Environment

Install the upstream training extra:

    python -m pip install -e "third_party/vggt-omega[train]"

VidEoMT additionally requires compatible Detectron2 and SciPy dependencies. E3A pins NumPy below 2, so use a compatible SciPy release (for example, SciPy 1.13.x). Build Detectron2 against the target PyTorch/CUDA environment; do not install an arbitrary prebuilt wheel.

A fully runnable training config remains gated on raw dataset readers, action vocabulary generation, event-window sampling, coordinate-transform validation, and source-video-level split auditing.
