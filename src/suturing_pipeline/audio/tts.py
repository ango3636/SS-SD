from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence


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


def _synthesize_gtts(text: str, out_mp3: Path, lang: str = "en") -> None:
    try:
        from gtts import gTTS
    except ImportError as e:
        raise RuntimeError(
            "gTTS is required for narration synthesis. "
            "Install it with `pip install gTTS`."
        ) from e
    tts = gTTS(text=text, lang=lang)
    tts.save(str(out_mp3))


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
    provider: str = "gtts",
    voice: str = "en",
    min_segment_seconds: float = 0.35,
) -> Dict[str, object]:
    """Create a single timeline-aligned narration track from segment text."""
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

    with tempfile.TemporaryDirectory(prefix="jigsaws_tts_") as td:
        tmp_dir = Path(td)
        clip_paths: List[Path] = []
        delays_ms: List[int] = []
        total_end = 0.0
        for i, seg in enumerate(valid):
            start = max(0.0, float(seg["start_time"]))
            end = max(start + min_segment_seconds, float(seg["end_time"]))
            total_end = max(total_end, end)
            text = str(seg["narration_text"]).strip()
            clip_path = tmp_dir / f"seg_{i:04d}.mp3"
            if provider == "gtts":
                _synthesize_gtts(text, clip_path, lang=voice)
            else:
                raise RuntimeError(f"Unsupported TTS provider: {provider}")
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
            "anullsrc=channel_layout=stereo:sample_rate=24000",
        ]
        for clip in clip_paths:
            cmd.extend(["-i", str(clip)])
        cmd.extend(
            [
                "-filter_complex",
                _build_amix_filter(delays_ms),
                "-map",
                "[aout]",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
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
