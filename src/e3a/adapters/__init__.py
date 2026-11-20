"""Optional adapters to upstream tools used for baselines and supervision."""

from e3a.adapters.sam3_video import SAM31VideoTeacher
from e3a.adapters.videomt import VidEoMTMaskCriterion

__all__ = ["SAM31VideoTeacher", "VidEoMTMaskCriterion"]
