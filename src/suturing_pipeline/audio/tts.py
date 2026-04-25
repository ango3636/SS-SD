"""Bark-backed text-to-speech for JIGSAWS narration.

Replaces the previous gTTS implementation. Exposes three layers:

1. :class:`BarkTTSConverter` - lazy-loaded Bark model wrapper with the two
   methods the compositor calls directly:

   * ``narration_to_audio(text, out_wav=..., target_duration_sec=...)`` -
     synthesises one segment as a 24 kHz mono ``numpy`` waveform,
     optionally time-stretched with ``librosa`` to fit a target duration
     and optionally written to ``.wav`` via ``scipy.io.wavfile.write``.
   * ``build_trial_audio_track(segments, output_audio_path, ...)`` -
     places per-segment waveforms onto a single silent trial timeline
     keyed on each segment's ``start_time`` / ``end_time``.

2. Module-level compatibility shims (``synthesize_narration_audio`` /
   ``mux_audio_to_video``) so existing callers - notably
   ``scripts/generate_eval_video.py`` and ``scripts/streamlit_compare.py``
   - keep working with the same signatures they used under gTTS.

3. ``mux_audio_to_video`` - ffmpeg mux helper (unchanged behaviour).

The Bark model and HuggingFace dependencies are imported lazily inside
:class:`BarkTTSConverter.__init__` so importing this module is cheap and
safe for unit tests that never build the converter.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

BARK_SAMPLE_RATE = 24_000
DEFAULT_VOICE_PRESET = "v2/en_speaker_6"


def _resolve_ffmpeg() -> Optional[str]:
    from shutil import which

    ffmpeg = which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _write_wav_int16(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Write a ``[-1, 1]`` float waveform to 16-bit PCM WAV via scipy."""
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    wav_int16 = (wav * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, wav_int16)


class BarkTTSConverter:
    """Lazy wrapper around ``suno/bark`` for clinical-narration TTS.

    The model (~6GB) is only downloaded / loaded when the converter is
    instantiated, so pure-unit tests that never call Bark stay cheap.
    ``voice_preset`` defaults to ``v2/en_speaker_6`` which reads as a
    calm male clinical-instructor voice.
    """

    def __init__(
        self,
        voice_preset: str = DEFAULT_VOICE_PRESET,
        device: Optional[str] = None,
        model_id: str = "suno/bark",
    ) -> None:
        from transformers import AutoProcessor, BarkModel
        import torch

        self.model_id = model_id
        self.voice_preset = voice_preset
        self.device = _pick_device(device)
        self.sample_rate = BARK_SAMPLE_RATE
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = BarkModel.from_pretrained(model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def _generate_waveform(self, text: str, voice_preset: str) -> np.ndarray:
        """Run Bark once and return a 1-D float32 waveform at 24 kHz."""
        inputs = self.processor(text, voice_preset=voice_preset)
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with self._torch.inference_mode():
            audio_tensor = self.model.generate(**inputs, do_sample=True)
        wav = audio_tensor.detach().cpu().numpy().squeeze().astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        # Bark occasionally returns slightly >1 peaks; normalise defensively.
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        return wav

    @staticmethod
    def _time_stretch(
        waveform: np.ndarray, sample_rate: int, target_duration_sec: float
    ) -> np.ndarray:
        """Use librosa.effects.time_stretch to fit ``target_duration_sec``.

        The stretch rate is ``src_duration / target_duration``
        (>1 = speed up, <1 = slow down).  librosa is lazy-imported so the
        rest of the module stays importable without the library.
        """
        if waveform.size == 0 or target_duration_sec <= 0:
            return waveform
        src_duration = waveform.shape[0] / float(sample_rate)
        if src_duration <= 0:
            return waveform
        rate = src_duration / float(target_duration_sec)
        # Skip the operation for near-1.0 rates (~3% tolerance) because
        # phase-vocoder stretching is not free and adds artefacts.
        if abs(rate - 1.0) < 0.03:
            return waveform
        import librosa

        stretched = librosa.effects.time_stretch(
            y=np.asarray(waveform, dtype=np.float32), rate=float(rate)
        )
        return stretched.astype(np.float32)

    def narration_to_audio(
        self,
        text: str,
        out_wav: Optional[str | Path] = None,
        target_duration_sec: Optional[float] = None,
        voice_preset: Optional[str] = None,
    ) -> np.ndarray:
        """Synthesise ``text`` with Bark; optionally stretch + save .wav.

        Returns the float32 waveform at :data:`BARK_SAMPLE_RATE`.
        When ``out_wav`` is set the waveform is also written as 16-bit PCM
        via ``scipy.io.wavfile.write``.  When ``target_duration_sec`` is
        set the raw Bark output is time-stretched with librosa before
        returning/saving so it slots into a gesture's on-screen duration.
        """
        preset = voice_preset or self.voice_preset
        wav = self._generate_waveform(text, voice_preset=preset)
        if target_duration_sec is not None:
            wav = self._time_stretch(wav, self.sample_rate, target_duration_sec)
        if out_wav is not None:
            _write_wav_int16(Path(out_wav), wav, self.sample_rate)
        return wav

    def build_trial_audio_track(
        self,
        segments: Sequence[Dict[str, object]],
        output_audio_path: str | Path,
        min_segment_seconds: float = 0.35,
        voice_preset: Optional[str] = None,
    ) -> Dict[str, object]:
        """Build one trial-length WAV from narration segments.

        ``segments`` mirrors the existing ``synthesize_narration_audio``
        shape: each dict must carry ``narration_text``, ``start_time``,
        ``end_time``.  For each segment we synthesise with Bark,
        time-stretch to fit ``end_time - start_time``, and place the
        clip at ``start_time`` in a zero-initialised buffer sized to
        cover ``max(end_time)``.  Overlapping segments are summed then
        peak-normalised.
        """
        valid = [
            s
            for s in segments
            if str(s.get("narration_text", "")).strip()
            and float(s.get("end_time", 0.0)) > float(s.get("start_time", 0.0))
        ]
        if not valid:
            raise RuntimeError(
                "No narration segments with valid text/time were provided."
            )
        preset = voice_preset or self.voice_preset
        total_end = max(float(s["end_time"]) for s in valid)
        total_samples = int(round((total_end + 0.25) * self.sample_rate))
        track = np.zeros(total_samples, dtype=np.float32)

        for seg in valid:
            start = max(0.0, float(seg["start_time"]))
            end = max(start + min_segment_seconds, float(seg["end_time"]))
            duration = end - start
            text = str(seg["narration_text"]).strip()
            wav = self._generate_waveform(text, voice_preset=preset)
            wav = self._time_stretch(wav, self.sample_rate, duration)
            start_sample = int(round(start * self.sample_rate))
            end_sample = min(start_sample + wav.shape[0], total_samples)
            if end_sample <= start_sample:
                continue
            slice_len = end_sample - start_sample
            track[start_sample:end_sample] += wav[:slice_len]

        peak = float(np.max(np.abs(track))) if track.size else 0.0
        if peak > 1.0:
            track = track / peak

        out_path = Path(output_audio_path)
        _write_wav_int16(out_path, track, self.sample_rate)
        return {
            "audio_path": str(out_path),
            "provider": "bark",
            "voice": preset,
            "sample_rate": self.sample_rate,
            "segments": len(valid),
            "duration_seconds": round(total_samples / self.sample_rate, 3),
        }


_DEFAULT_CONVERTER: Optional[BarkTTSConverter] = None


def _get_converter(
    voice_preset: str = DEFAULT_VOICE_PRESET,
    device: Optional[str] = None,
) -> BarkTTSConverter:
    """Lazy singleton so repeat calls in one process don't reload Bark."""
    global _DEFAULT_CONVERTER
    if (
        _DEFAULT_CONVERTER is None
        or _DEFAULT_CONVERTER.voice_preset != voice_preset
        or (device is not None and _DEFAULT_CONVERTER.device != device)
    ):
        _DEFAULT_CONVERTER = BarkTTSConverter(
            voice_preset=voice_preset, device=device
        )
    return _DEFAULT_CONVERTER


def synthesize_narration_audio(
    segments: Sequence[Dict[str, object]],
    output_audio_path: str | Path,
    provider: str = "bark",
    voice: str = DEFAULT_VOICE_PRESET,
    min_segment_seconds: float = 0.35,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Back-compat entry point used by ``generate_eval_video.py``.

    Same signature as the original gTTS implementation so callers don't
    break; only Bark is supported going forward.  ``voice`` now takes a
    Bark voice-preset string instead of a language code.
    """
    if provider.lower() != "bark":
        raise RuntimeError(
            f"Unsupported TTS provider {provider!r}; this build only ships "
            "Bark (suno/bark). Remove --tts_provider or pass --tts_provider bark."
        )
    converter = _get_converter(voice_preset=voice, device=device)
    info = converter.build_trial_audio_track(
        segments=segments,
        output_audio_path=output_audio_path,
        min_segment_seconds=min_segment_seconds,
    )
    return info


def mux_audio_to_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> None:
    """Mux an audio track onto a video via ffmpeg (video copy, audio AAC)."""
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to mux narrated videos.")
    v_path = Path(video_path)
    a_path = Path(audio_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(v_path),
        "-i",
        str(a_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.exists():
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Failed to mux audio/video (rc={proc.returncode}). {stderr}"
        )


__all__ = [
    "BARK_SAMPLE_RATE",
    "DEFAULT_VOICE_PRESET",
    "BarkTTSConverter",
    "synthesize_narration_audio",
    "mux_audio_to_video",
]
