"""E3A data batching and source-annotation utilities."""

from e3a.data.actions import action_label_key, build_dataset_prefixed_action_vocabulary
from e3a.data.collate import collate_e3a_batch, collate_entity_targets
from e3a.data.egoexo import (
    build_egoexo_anticipation_target,
    build_egoexo_keystep_vocabulary,
    build_egoexo_relation_clip_targets,
    load_egoexo_keystep_segments,
    parse_egoexo_keystep_segments,
    load_egoexo_hand_pose_annotations,
    parse_egoexo_hand_pose_annotations,
    resolve_egoexo_video_path,
)
from e3a.data.hoi4d import (
    build_hoi4d_action_vocabulary,
    build_hoi4d_anticipation_target,
    build_hoi4d_entity_clip_targets,
    decode_hoi4d_motion_mask,
    hoi4d_color_palette,
    load_hoi4d_action_segments,
    parse_hoi4d_action_segments,
)
from e3a.data.hoi4d_dataset import (
    HOI4DDataset,
    HOI4DSequencePaths,
    collate_hoi4d_batch,
    make_hoi4d_dataloader,
    resolve_hoi4d_sequence_paths,
)

__all__ = [
    "action_label_key",
    "build_dataset_prefixed_action_vocabulary",
    "collate_entity_targets",
    "collate_e3a_batch",
    "hoi4d_color_palette",
    "decode_hoi4d_motion_mask",
    "build_hoi4d_entity_clip_targets",
    "parse_hoi4d_action_segments",
    "load_hoi4d_action_segments",
    "build_hoi4d_action_vocabulary",
    "build_hoi4d_anticipation_target",
    "HOI4DDataset",
    "HOI4DSequencePaths",
    "resolve_hoi4d_sequence_paths",
    "collate_hoi4d_batch",
    "make_hoi4d_dataloader",
    "parse_egoexo_hand_pose_annotations",
    "load_egoexo_hand_pose_annotations",
    "parse_egoexo_keystep_segments",
    "load_egoexo_keystep_segments",
    "build_egoexo_keystep_vocabulary",
    "build_egoexo_anticipation_target",
    "build_egoexo_relation_clip_targets",
    "resolve_egoexo_video_path",
]
