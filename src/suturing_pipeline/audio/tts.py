from __future__ import annotations

import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BARK_SAMPLE_RATE = 24000
_BARK_CACHE: Dict[str, object] = {}


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


def _load_bark() -> Tuple[object, object, object]:
    """Load Bark processor + model once and cache them.

    Bark is ~4GB; loading per-segment would dominate wall-clock. We cache
    the processor, model, and torch device in a module-level dict.
    """
    if "processor" in _BARK_CACHE and "model" in _BARK_CACHE:
        return (
            _BARK_CACHE["processor"],
            _BARK_CACHE["model"],
            _BARK_CACHE["device"],
        )
    try:
        import torch
        from transformers import AutoProcessor, BarkModel
    except ImportError as e:
        raise RuntimeError(
            "Bark requires transformers>=4.31.0 and torch. "
            "Install via `pip install -r requirements.txt`."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn(
            "Bark is running on CPU; expect ~30x slower synthesis than CUDA. "
            "Install CUDA torch to use a GPU.",
            RuntimeWarning,
            stacklevel=2,
        )

    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark").to(device)
    model.eval()

    _BARK_CACHE["processor"] = processor
    _BARK_CACHE["model"] = model
    _BARK_CACHE["device"] = device
    return processor, model, device


def _synthesize_bark(
    text: str,
    out_wav: Path,
    voice_preset: str = "v2/en_speaker_6",
    window_seconds: Optional[float] = None,
) -> float:
    """Synthesize ``text`` with Bark and write a WAV at 24kHz.

    If ``window_seconds`` is provided and the natural clip is longer than
    the window, the waveform is time-stretched (hybrid-stretch) via
    ``librosa.effects.time_stretch`` so it fits the gesture window.

    Returns the final duration in seconds.
    """
    import scipy.io.wavfile as wavfile
    import torch

    processor, model, device = _load_bark()

    inputs = processor(text, voice_preset=voice_preset)
    inputs = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        audio_array = model.generate(**inputs)

    audio = np.squeeze(audio_array.cpu().numpy()).astype(np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    natural_dur = len(audio) / float(BARK_SAMPLE_RATE)
    if (
        window_seconds is not None
        and window_seconds > 0
        and natural_dur > window_seconds
    ):
        try:
            import librosa

            rate = float(natural_dur / window_seconds)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        except ImportError as e:
            raise RuntimeError(
                "librosa is required for hybrid time-stretch fitting. "
                "Install via `pip install librosa`."
            ) from e

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    wavfile.write(str(out_wav), BARK_SAMPLE_RATE, pcm)
    return len(pcm) / float(BARK_SAMPLE_RATE)


def _build_amix_filter(delays_ms: Sequence[int]) -> str:
    if not delays_ms:
        return "[0:a]anull[aout]"
    parts: List[str] = []
    for i, delay in enumerate(delays_ms, start=1):
        parts.append(f"[{i}:a]adelay={delay}|{delay}[a{i}]")
    mix_inputs = "".join(["[0:a]"] + [f"[a{i}]" for i in range(1, len(delays_ms) + 1)])
    parts.append(
        f"{mix_inputs}amix=inputs={len(delays_ms) + 1}:duration=longest:normalize=0[aout]"
    )
    return ";".join(parts)


def synthesize_narration_audio(
    segments: Sequence[Dict[str, object]],
    output_audio_path: str | Path,
    provider: str = "bark",
    voice: str = "v2/en_speaker_6",
    min_segment_seconds: float = 0.35,
) -> Dict[str, object]:
    """Create a single timeline-aligned narration track from segment text.

    ``voice`` is forwarded to Bark as the voice preset (e.g.
    ``v2/en_speaker_6``). ``provider`` is kept for call-site stability and
    must be ``"bark"``.
    """
    if provider != "bark":
        raise RuntimeError(f"Unsupported TTS provider: {provider!r}")

    out_path = Path(output_audio_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required for narration alignment/muxing. "
            "Install ffmpeg or imageio-ffmpeg."
        )

    valid = [
        s
        for s in segments
        if str(s.get("narration_text", "")).strip()
        and float(s.get("end_time", 0.0)) > float(s.get("start_time", 0.0))
    ]
    if not valid:
        raise RuntimeError("No narration segments with valid text/time were provided.")

    # .wav output uses PCM; other extensions fall back to AAC.
    out_ext = out_path.suffix.lower()
    if out_ext == ".wav":
        audio_codec_args = ["-c:a", "pcm_s16le"]
    else:
        audio_codec_args = ["-c:a", "aac", "-b:a", "128k"]

    with tempfile.TemporaryDirectory(prefix="jigsaws_tts_") as td:
        tmp_dir = Path(td)
        clip_paths: List[Path] = []
        delays_ms: List[int] = []
        total_end = 0.0
        for i, seg in enumerate(valid):
            start = max(0.0, float(seg["start_time"]))
            nominal_end = float(seg["end_time"])
            next_start = (
                float(valid[i + 1]["start_time"])
                if i + 1 < len(valid)
                else float("inf")
            )
            # min_segment_seconds lengthens the TTS window, but delays are still
            # tied to real gesture start_time. Without capping ``end`` at the
            # next segment's start, clips overlap in ffmpeg amix (double voice).
            end = max(start + min_segment_seconds, nominal_end)
            end = min(end, next_start)
            if end <= start:
                end = min(nominal_end, next_start)
            total_end = max(total_end, end, nominal_end)
            text = str(seg["narration_text"]).strip()
            clip_path = tmp_dir / f"seg_{i:04d}.wav"
            _synthesize_bark(
                text=text,
                out_wav=clip_path,
                voice_preset=voice,
                window_seconds=end - start,
            )
            clip_paths.append(clip_path)
            delays_ms.append(int(round(start * 1000.0)))

        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-t",
            f"{total_end + 0.5:.3f}",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={BARK_SAMPLE_RATE}",
        ]
        for clip in clip_paths:
            cmd.extend(["-i", str(clip)])
        cmd.extend(
            [
                "-filter_complex",
                _build_amix_filter(delays_ms),
                "-map",
                "[aout]",
                *audio_codec_args,
                str(out_path),
            ]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_path.exists():
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"Failed to build narration track (rc={proc.returncode}). {stderr}"
            )

    return {
        "audio_path": str(out_path),
        "provider": provider,
        "voice": voice,
        "segments": len(valid),
    }


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
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.exists():
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Failed to mux audio/video (rc={proc.returncode}). {stderr}")
