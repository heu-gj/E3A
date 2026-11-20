"""E3A data batching and source-annotation utilities."""

from e3a.data.collate import collate_e3a_batch, collate_entity_targets
from e3a.data.egoexo import (
    build_egoexo_relation_clip_targets,
    load_egoexo_hand_pose_annotations,
    parse_egoexo_hand_pose_annotations,
    resolve_egoexo_video_path,
)
from e3a.data.hoi4d import (
    build_hoi4d_entity_clip_targets,
    decode_hoi4d_motion_mask,
    hoi4d_color_palette,
)

__all__ = [
    "collate_entity_targets",
    "collate_e3a_batch",
    "hoi4d_color_palette",
    "decode_hoi4d_motion_mask",
    "build_hoi4d_entity_clip_targets",
    "parse_egoexo_hand_pose_annotations",
    "load_egoexo_hand_pose_annotations",
    "build_egoexo_relation_clip_targets",
    "resolve_egoexo_video_path",
]
