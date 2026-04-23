"""Audio and narration helpers for surgical-video generation."""

from .narration_templates import (
    GESTURE_DESCRIPTIONS,
    build_expert_speed_stats,
    build_narration_payload,
    collapse_frame_records,
    extract_kinematic_summary,
)
from .tts import mux_audio_to_video, synthesize_narration_audio

__all__ = [
    "GESTURE_DESCRIPTIONS",
    "extract_kinematic_summary",
    "build_expert_speed_stats",
    "build_narration_payload",
    "collapse_frame_records",
    "synthesize_narration_audio",
    "mux_audio_to_video",
]
