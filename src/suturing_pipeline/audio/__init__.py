"""Audio and narration helpers for surgical-video generation."""

from .narration_templates import (
    GESTURE_DESCRIPTIONS,
    build_expert_speed_stats,
    build_narration_payload,
    collapse_frame_records,
    extract_kinematic_summary,
)
from .tts import (
    BARK_SAMPLE_RATE,
    DEFAULT_VOICE_PRESET,
    BarkTTSConverter,
    mux_audio_to_video,
    synthesize_narration_audio,
)
from .compositor import (
    generate_shared_audio_track,
    mux_audio_to_generated_video,
    mux_audio_to_raw_video,
    side_by_side_comparison,
)

__all__ = [
    "GESTURE_DESCRIPTIONS",
    "extract_kinematic_summary",
    "build_expert_speed_stats",
    "build_narration_payload",
    "collapse_frame_records",
    "BARK_SAMPLE_RATE",
    "DEFAULT_VOICE_PRESET",
    "BarkTTSConverter",
    "synthesize_narration_audio",
    "mux_audio_to_video",
    "generate_shared_audio_track",
    "mux_audio_to_raw_video",
    "mux_audio_to_generated_video",
    "side_by_side_comparison",
]
