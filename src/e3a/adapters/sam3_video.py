"""Thin adapter for the official SAM 3.1 multiplex video predictor.

SAM 3.1 is an optional, independent teacher/baseline. It is not loaded by the
E3A model's normal forward pass.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Iterator


class SAM31VideoTeacher:
    """Wrap the official SAM 3.1 predictor request API.

    The predictor is built by the upstream recommended
    build_sam3_multiplex_video_predictor entry point. A predictor can be
    injected for tests or for externally managed inference.

    SAM 3.1's current upstream installation guide specifies Python 3.12+,
    PyTorch 2.10+, and CUDA 12.8+. Checkpoint access may require Hugging Face
    approval and authentication.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | PathLike[str] | None = None,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        use_fa3: bool = False,
        use_rope_real: bool = True,
        compile: bool = False,
        warm_up: bool = False,
        predictor: Any | None = None,
    ) -> None:
        if predictor is None:
            try:
                from sam3.model_builder import build_sam3_multiplex_video_predictor
            except ImportError as exc:
                raise ImportError(
                    "SAM 3 is not installed. Follow the official SAM 3 "
                    "environment instructions, then run "
                    "pip install -e third_party/sam3."
                ) from exc

            kwargs: dict[str, Any] = {
                "max_num_objects": max_num_objects,
                "multiplex_count": multiplex_count,
                "use_fa3": use_fa3,
                "use_rope_real": use_rope_real,
                "compile": compile,
                "warm_up": warm_up,
            }
            if checkpoint_path is not None:
                kwargs["checkpoint_path"] = str(Path(checkpoint_path).expanduser())

            predictor = build_sam3_multiplex_video_predictor(**kwargs)

        self.predictor = predictor

    def start_session(
        self,
        resource_path: str | PathLike[str],
        **request_options: Any,
    ) -> dict[str, Any]:
        """Start an upstream SAM 3.1 video session."""
        request = {
            "type": "start_session",
            "resource_path": str(Path(resource_path).expanduser()),
            **request_options,
        }
        return self.predictor.handle_request(request)

    def add_text_prompt(
        self,
        session_id: str,
        frame_index: int,
        text: str,
        **request_options: Any,
    ) -> dict[str, Any]:
        """Add an upstream text prompt to a video frame."""
        if not text.strip():
            raise ValueError("SAM 3.1 text prompt must be non-empty.")

        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": text,
            **request_options,
        }
        return self.predictor.handle_request(request)

    def propagate(
        self,
        session_id: str,
        *,
        direction: str = "forward",
        **request_options: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield official SAM 3.1 propagated frame outputs."""
        if direction not in {"forward", "backward", "both"}:
            raise ValueError(
                "direction must be one of 'forward', 'backward', or 'both'."
            )

        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            **request_options,
        }
        yield from self.predictor.handle_stream_request(request)

    def close_session(self, session_id: str) -> dict[str, Any]:
        """Close an upstream SAM 3.1 session."""
        return self.predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )
