"""Audio and narration helpers for surgical-video generation."""

from .llm_narration import (
    DEFAULT_HF_NARRATION_MODEL,
    DEFAULT_OLLAMA_BASE,
    DEFAULT_OLLAMA_MODEL,
    apply_llm_narration_to_segments,
    synthesize_narration_line,
)
from .foley import FoleyLibrary
from .narration_templates import (
    GESTURE_AMBIENCE,
    GESTURE_DESCRIPTIONS,
    build_expert_speed_stats,
    build_llm_prompt,
    build_narration_payload,
    collapse_frame_records,
    extract_kinematic_summary,
    max_narration_words_for_duration,
)
from .tts import (
    BARK_SAMPLE_RATE,
    DEFAULT_VOICE_PRESET,
    BarkTTSConverter,
    ORAmbienceGenerator,
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
    "DEFAULT_HF_NARRATION_MODEL",
    "DEFAULT_OLLAMA_BASE",
    "DEFAULT_OLLAMA_MODEL",
    "apply_llm_narration_to_segments",
    "synthesize_narration_line",
    "GESTURE_DESCRIPTIONS",
    "FoleyLibrary",
    "GESTURE_AMBIENCE",
    "extract_kinematic_summary",
    "build_expert_speed_stats",
    "build_narration_payload",
    "build_llm_prompt",
    "max_narration_words_for_duration",
    "collapse_frame_records",
    "BARK_SAMPLE_RATE",
    "DEFAULT_VOICE_PRESET",
    "BarkTTSConverter",
    "ORAmbienceGenerator",
    "synthesize_narration_audio",
    "mux_audio_to_video",
    "generate_shared_audio_track",
    "mux_audio_to_raw_video",
    "mux_audio_to_generated_video",
    "side_by_side_comparison",
]
