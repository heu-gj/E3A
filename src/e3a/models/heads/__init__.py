"""Task-specific E3A prediction heads."""

from e3a.models.heads.anticipation_head import AnticipationHead
from e3a.models.heads.entity_mask_head import EntityMaskHead
from e3a.models.heads.geometry_head import GeometryHead
from e3a.models.heads.hand_head import HandPoseHead
from e3a.models.heads.interaction_head import InteractionHead

__all__ = [
    "EntityMaskHead",
    "GeometryHead",
    "HandPoseHead",
    "InteractionHead",
    "AnticipationHead",
]
