"""Back-compat re-exports for Bark TTS and mux helpers.

Implementation lives in :mod:`suturing_pipeline.audio.tts_converter`.
"""

from .foley import FoleyLibrary
from .tts_converter import (
    BARK_SAMPLE_RATE,
    DEFAULT_VOICE_PRESET,
    BarkTTSConverter,
    ORAmbienceGenerator,
    mux_audio_to_video,
    synthesize_narration_audio,
)

__all__ = [
    "BARK_SAMPLE_RATE",
    "DEFAULT_VOICE_PRESET",
    "BarkTTSConverter",
    "FoleyLibrary",
    "ORAmbienceGenerator",
    "synthesize_narration_audio",
    "mux_audio_to_video",
]
