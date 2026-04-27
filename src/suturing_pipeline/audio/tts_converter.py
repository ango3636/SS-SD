"""Bark TTS, duration fitting (capped librosa stretch + tail-silence trim), and OR ambience.

``facebook/audiogen-medium`` (via audiocraft) is optional: only loaded when
:class:`ORAmbienceGenerator` is used.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

from .foley import FoleyLibrary, place_foley_on_segment
from .narration_templates import GESTURE_AMBIENCE

BARK_SAMPLE_RATE = 24_000
DEFAULT_VOICE_PRESET = "v2/en_speaker_9"

# librosa.effects.time_stretch ``rate``: >1 speeds up (shortens), <1 slows down.
_MIN_STRETCH_RATE = 0.8
_MAX_STRETCH_RATE = 1.3
# Ambience sits ~18 dB below the Bark narration peak before final trial normalisation.
_OR_AMBIENCE_DB_UNDER_NARRATION = -18.0


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
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    wav_int16 = (wav * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, wav_int16)


def _trim_trailing_silence(
    waveform: np.ndarray, sample_rate: int, top_db: float = 45.0
) -> np.ndarray:
    """Drop low-energy samples from the end of ``waveform`` (mono float32)."""
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.size == 0:
        return wav
    import librosa

    rev = wav[::-1].copy()
    trimmed, _ = librosa.effects.trim(rev, top_db=top_db)
    out = trimmed[::-1]
    if out.size == 0:
        return wav
    return out.astype(np.float32)


def _ambience_gain_linear() -> float:
    return float(10.0 ** (_OR_AMBIENCE_DB_UNDER_NARRATION / 20.0))


def _resample_waveform(
    waveform: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(waveform, dtype=np.float32)
    import librosa

    return librosa.resample(
        np.asarray(waveform, dtype=np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
    ).astype(np.float32)


def _schedule_narration_segments(
    segments: Sequence[Dict[str, object]],
    min_segment_seconds: float,
) -> List[tuple[float, float, Dict[str, object]]]:
    """Sort by timeline and delay starts so no two clips share the mix bus.

    JIGSAWS (or collapsed eval) segments can overlap in ``[start_time, end_time]``
    when annotations are dense or inconsistent; stacking Bark on the same samples
    makes speech unintelligible. This keeps order, preserves each segment's
    duration where possible, and pushes a segment forward only when it would
    otherwise overlap the previous *placed* end.
    """
    valid = [
        s
        for s in segments
        if str(s.get("narration_text", "")).strip()
        and float(s.get("end_time", 0.0)) > float(s.get("start_time", 0.0))
    ]
    if not valid:
        return []
    ordered = sorted(
        valid,
        key=lambda s: (float(s["start_time"]), float(s["end_time"])),
    )
    scheduled: List[tuple[float, float, Dict[str, object]]] = []
    mix_cursor = 0.0
    for seg in ordered:
        raw_s = max(0.0, float(seg["start_time"]))
        raw_e = float(seg["end_time"])
        raw_e = max(raw_s + float(min_segment_seconds), raw_e)
        start = max(raw_s, mix_cursor)
        end = max(start + float(min_segment_seconds), raw_e)
        if end <= start + 1e-9:
            continue
        scheduled.append((start, end, seg))
        mix_cursor = end
    return scheduled


def _match_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    if w.size == target_len:
        return w
    if w.size > target_len:
        return w[:target_len].copy()
    out = np.zeros(target_len, dtype=np.float32)
    out[: w.size] = w
    return out


class ORAmbienceGenerator:
    """Text-conditioned OR background audio via ``facebook/audiogen-medium`` (audiocraft)."""

    def __init__(
        self,
        model_id: str = "facebook/audiogen-medium",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = _pick_device(device)
        self._model = None
        self._native_sr: int = 16_000

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from audiocraft.models import AudioGen
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OR ambience needs the audiocraft package (AudioGen). "
                "Install with: pip install 'git+https://github.com/facebookresearch/audiocraft.git'"
            ) from e
        import torch

        model = AudioGen.get_pretrained(self.model_id, device=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        sr = getattr(model, "sample_rate", None)
        self._native_sr = int(sr) if sr is not None else 16_000

    def generate_for_prompt(
        self, prompt: str, duration_seconds: float, seed: Optional[int] = None
    ) -> np.ndarray:
        """Return mono float32 audio at :data:`BARK_SAMPLE_RATE` (~``duration_seconds`` long)."""
        self._ensure_model()
        assert self._model is not None
        dur = max(1.0, min(30.0, float(duration_seconds)))
        gen_sec = float(math.ceil(dur))
        self._model.set_generation_params(duration=gen_sec)
        if seed is not None:
            self._torch.manual_seed(int(seed))
        descriptions = [prompt]
        with self._torch.inference_mode():
            batch = self._model.generate(descriptions)
        if isinstance(batch, (list, tuple)):
            one = batch[0]
        else:
            one = batch
        if hasattr(one, "detach"):
            one = one.detach().cpu().float().numpy()
        wav = np.asarray(one, dtype=np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        wav = _resample_waveform(wav, orig_sr=self._native_sr, target_sr=BARK_SAMPLE_RATE)
        target_len = int(round(float(duration_seconds) * BARK_SAMPLE_RATE))
        return _match_length(wav, max(1, target_len))


class BarkTTSConverter:
    """Lazy wrapper around ``suno/bark`` for clinical-narration TTS."""

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
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        return wav

    @staticmethod
    def _fit_duration_with_stretch_trim(
        waveform: np.ndarray, sample_rate: int, target_duration_sec: float
    ) -> np.ndarray:
        """Stretch toward ``target_duration_sec`` with capped librosa rate; trim tail silence when rate would exceed cap."""
        if waveform.size == 0 or target_duration_sec <= 0:
            return waveform
        wav = np.asarray(waveform, dtype=np.float32).copy()
        target = float(target_duration_sec)

        while True:
            src_duration = wav.shape[0] / float(sample_rate)
            if src_duration <= 0:
                return wav
            rate = src_duration / target
            if rate <= _MAX_STRETCH_RATE + 1e-6:
                break
            trimmed = _trim_trailing_silence(wav, sample_rate)
            if trimmed.shape[0] >= wav.shape[0]:
                break
            wav = trimmed

        src_duration = wav.shape[0] / float(sample_rate)
        if src_duration <= 0:
            return wav
        rate = src_duration / target
        rate = max(_MIN_STRETCH_RATE, min(_MAX_STRETCH_RATE, rate))
        if abs(rate - 1.0) < 0.03:
            return wav
        import librosa

        stretched = librosa.effects.time_stretch(
            y=np.asarray(wav, dtype=np.float32), rate=float(rate)
        )
        return stretched.astype(np.float32)

    def narration_to_audio(
        self,
        text: str,
        out_wav: Optional[str | Path] = None,
        target_duration_sec: Optional[float] = None,
        voice_preset: Optional[str] = None,
    ) -> np.ndarray:
        preset = voice_preset or self.voice_preset
        wav = self._generate_waveform(text, voice_preset=preset)
        if target_duration_sec is not None:
            wav = self._fit_duration_with_stretch_trim(
                wav, self.sample_rate, target_duration_sec
            )
        if out_wav is not None:
            _write_wav_int16(Path(out_wav), wav, self.sample_rate)
        return wav

    def build_trial_audio_track(
        self,
        segments: Sequence[Dict[str, object]],
        output_audio_path: str | Path,
        min_segment_seconds: float = 0.35,
        voice_preset: Optional[str] = None,
        or_ambience: bool = False,
        ambience_generator: Optional[ORAmbienceGenerator] = None,
        foley_dir: Optional[str | Path] = None,
        foley_gain_db: float = -12.0,
        foley_align: Literal["start", "center"] = "start",
        foley_library: Optional[FoleyLibrary] = None,
    ) -> Dict[str, object]:
        scheduled = _schedule_narration_segments(segments, min_segment_seconds)
        if not scheduled:
            raise RuntimeError(
                "No narration segments with valid text/time were provided."
            )
        preset = voice_preset or self.voice_preset
        total_end = max(end for _, end, _ in scheduled)
        total_samples = int(round((total_end + 0.25) * self.sample_rate))
        track = np.zeros(total_samples, dtype=np.float32)

        amb_gen = None
        if or_ambience:
            amb_gen = ambience_generator or ORAmbienceGenerator(device=self.device)

        amb_gain = _ambience_gain_linear()

        foley_lib: Optional[FoleyLibrary] = None
        if foley_library is not None:
            foley_lib = foley_library
        elif foley_dir:
            cand = FoleyLibrary(foley_dir, sample_rate=self.sample_rate)
            foley_lib = cand if cand.enabled else None

        foley_segments = 0

        for start, end, seg in scheduled:
            duration = end - start
            text = str(seg["narration_text"]).strip()
            wav = self._generate_waveform(text, voice_preset=preset)
            wav = self._fit_duration_with_stretch_trim(
                wav, self.sample_rate, duration
            )
            if amb_gen is not None:
                gesture = str(seg.get("gesture", "")).strip() or "G1"
                amb_prompt = GESTURE_AMBIENCE.get(
                    gesture, GESTURE_AMBIENCE.get("DEFAULT", "")
                )
                amb = amb_gen.generate_for_prompt(amb_prompt, duration_seconds=duration)
                amb = _match_length(amb, wav.shape[0])
                wav = wav + amb.astype(np.float32) * amb_gain
            if foley_lib is not None:
                gesture = str(seg.get("gesture", "")).strip() or "G1"
                foley_wav = foley_lib.get_mono(gesture)
                if foley_wav is not None:
                    placed = place_foley_on_segment(
                        foley_wav,
                        wav.shape[0],
                        align=foley_align,
                    )
                    fg = float(10.0 ** (foley_gain_db / 20.0))
                    wav = wav + placed * fg
                    foley_segments += 1
            # Exact slot length prevents librosa stretch drift from bleeding into the
            # next segment (overlapping Bark lines).
            slot_samples = max(1, int(round(float(duration) * self.sample_rate)))
            wav = _match_length(wav, slot_samples)
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
            "segments": len(scheduled),
            "duration_seconds": round(total_samples / self.sample_rate, 3),
            "or_ambience": bool(or_ambience),
            "foley_dir": str(foley_lib.root) if foley_lib is not None else None,
            "foley_gain_db": float(foley_gain_db) if foley_lib is not None else None,
            "foley_align": str(foley_align) if foley_lib is not None else None,
            "foley_segments": int(foley_segments),
        }


_DEFAULT_CONVERTER: Optional[BarkTTSConverter] = None


def _get_converter(
    voice_preset: str = DEFAULT_VOICE_PRESET,
    device: Optional[str] = None,
) -> BarkTTSConverter:
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
    or_ambience: bool = False,
    foley_dir: Optional[str | Path] = None,
    foley_gain_db: float = -12.0,
    foley_align: Literal["start", "center"] = "start",
    foley_library: Optional[FoleyLibrary] = None,
) -> Dict[str, object]:
    if provider.lower() != "bark":
        raise RuntimeError(
            f"Unsupported TTS provider {provider!r}; this build only ships "
            "Bark (suno/bark). Remove --tts_provider or pass --tts_provider bark."
        )
    converter = _get_converter(voice_preset=voice, device=device)
    return converter.build_trial_audio_track(
        segments=segments,
        output_audio_path=output_audio_path,
        min_segment_seconds=min_segment_seconds,
        or_ambience=or_ambience,
        foley_dir=foley_dir,
        foley_gain_db=foley_gain_db,
        foley_align=foley_align,
        foley_library=foley_library,
    )


def mux_audio_to_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> None:
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
    "ORAmbienceGenerator",
    "FoleyLibrary",
    "synthesize_narration_audio",
    "mux_audio_to_video",
]
