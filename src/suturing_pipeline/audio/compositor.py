"""Dual-audio compositor: shared Bark narration + raw/generated muxing.

The narration track is derived once from a trial's kinematics +
transcription (30 Hz) and then muxed onto:

1. the original JIGSAWS ``capture*.avi``, producing
   ``<trial>_raw_narrated.mp4``,
2. an SD-generated video (either an MP4 or a directory of PNG frames),
   producing ``<trial>_generated_narrated.mp4``,

and finally stitched into one ``<trial>_comparison.mp4`` with "Original"
and "Generated" text overlays via :mod:`moviepy`.

moviepy 2.x is required (``from moviepy import VideoFileClip``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from suturing_pipeline.audio.llm_narration import (
    DEFAULT_HF_NARRATION_MODEL,
    DEFAULT_OLLAMA_BASE,
    DEFAULT_OLLAMA_MODEL,
    apply_llm_narration_to_segments,
)
from suturing_pipeline.audio.narration_templates import (
    GESTURE_DESCRIPTIONS,
    build_expert_speed_stats,
    build_narration_payload,
)
from suturing_pipeline.audio.tts import (
    DEFAULT_VOICE_PRESET,
    BarkTTSConverter,
)
from suturing_pipeline.data.data_utils import (
    parse_kinematics,
    parse_transcription,
)

JIGSAWS_SOURCE_FPS = 30.0
# JIGSAWS uses Title-cased task names on disk even though CLI args are
# often lowercase; keep one canonical mapping.
_TASK_PREFIX = {"suturing": "Suturing", "knot_tying": "Knot_Tying", "needle_passing": "Needle_Passing"}


def _resolve_ci(path: Path) -> Path:
    """Case-insensitive segment-wise path resolution (Linux-safe)."""
    parts = path.parts
    if not parts:
        return path
    cur = Path(parts[0])
    for seg in parts[1:]:
        direct = cur / seg
        if direct.exists():
            cur = direct
            continue
        try:
            matches = [p for p in cur.iterdir() if p.name.lower() == seg.lower()]
        except OSError:
            return path
        if not matches:
            return path
        cur = matches[0]
    return cur


def _task_prefix(task: str) -> str:
    return _TASK_PREFIX.get(task.lower(), task[:1].upper() + task[1:])


def _locate_trial_assets(
    trial_name: str, task: str, data_root: str | Path
) -> Dict[str, Path]:
    """Find kinematics / transcription / capture1 for ``trial_name``.

    Returns a dict with keys ``kinematics``, ``transcription``, ``video``.
    Missing entries are still returned as best-guess paths; callers
    should call ``.exists()`` and surface a useful error.
    """
    root = Path(data_root)
    task_dir = _resolve_ci(root / task)
    prefix = _task_prefix(task)
    suffix = trial_name.replace(f"{prefix}_", "")
    kin = _resolve_ci(
        task_dir / "kinematics" / "allgestures" / f"{prefix}_{suffix}.txt"
    )
    tx = _resolve_ci(task_dir / "transcriptions" / f"{prefix}_{suffix}.txt")
    video_dir = _resolve_ci(task_dir / "video")
    video = video_dir / f"{prefix}_{suffix}_capture1.avi"
    if not video.exists():
        alt = video_dir / f"{prefix}_{suffix}_capture2.avi"
        if alt.exists():
            video = alt
    return {"kinematics": kin, "transcription": tx, "video": video}


def _build_segments_from_trial(
    kinematics: np.ndarray,
    transcription: List[tuple],
    expert_stats: Optional[Dict[str, Dict[str, float]]] = None,
    source_fps: float = JIGSAWS_SOURCE_FPS,
    min_count_for_empirical: int = 8,
) -> List[Dict[str, object]]:
    """Turn (start_frame, end_frame, gesture) segments into narration segments.

    The segment timeline is in *source* seconds (JIGSAWS is 30 Hz), so
    the resulting audio track matches the original ``capture*.avi`` one-
    to-one.
    """
    segments: List[Dict[str, object]] = []
    for start_f, end_f, gesture in transcription:
        s = max(0, int(start_f))
        e = min(int(end_f), kinematics.shape[0] - 1)
        if e < s:
            continue
        kin_segment = kinematics[s : e + 1]
        payload = build_narration_payload(
            gesture_label=gesture,
            kinematic_segment=kin_segment,
            expert_speed_stats=expert_stats,
            min_count_for_empirical=min_count_for_empirical,
        )
        segments.append(
            {
                "start_time": round(s / source_fps, 4),
                "end_time": round((e + 1) / source_fps, 4),
                "start_frame": s,
                "end_frame": e,
                "gesture": gesture,
                "gesture_description": GESTURE_DESCRIPTIONS.get(
                    gesture, f"Performing {gesture}"
                ),
                "narration_text": payload["narration_text"],
                "summary": payload["summary"],
            }
        )
    return segments


def generate_shared_audio_track(
    trial_name: str,
    task: str,
    data_root: str | Path,
    output_dir: str | Path,
    api_key: Optional[str] = None,  # reserved; use hf_token for HF narration
    voice_preset: str = DEFAULT_VOICE_PRESET,
    tts_converter: Optional[BarkTTSConverter] = None,
    device: Optional[str] = None,
    min_segment_seconds: float = 0.5,
    or_ambience: bool = False,
    narration_backend: str = "template",
    ollama_base_url: str = DEFAULT_OLLAMA_BASE,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    hf_narration_model: str = DEFAULT_HF_NARRATION_MODEL,
    hf_token: Optional[str] = None,
    narration_llm_timeout_sec: float = 120.0,
) -> str:
    """Build one Bark narration WAV covering the full JIGSAWS trial.

    When ``narration_backend`` is ``ollama`` or ``huggingface``, each segment's
    ``narration_text`` is produced by an LLM first (speech-first), then Bark
    synthesises audio. ``template`` keeps the built-in kinematic templates.

    The returned path is ``<output_dir>/<trial_name>_narration.wav``.
    """
    _ = api_key  # reserved
    assets = _locate_trial_assets(trial_name, task, data_root)
    if not assets["kinematics"].exists() or not assets["transcription"].exists():
        raise FileNotFoundError(
            f"Missing kinematics or transcription for trial {trial_name!r}. "
            f"Looked at: kinematics={assets['kinematics']}, "
            f"transcription={assets['transcription']}"
        )
    kinematics = parse_kinematics(assets["kinematics"])
    transcription = parse_transcription(assets["transcription"])
    try:
        expert_stats = build_expert_speed_stats(data_root, task=task)
    except Exception:  # pragma: no cover - defensive
        expert_stats = {}
    segments = _build_segments_from_trial(
        kinematics=kinematics,
        transcription=transcription,
        expert_stats=expert_stats,
    )
    if not segments:
        raise RuntimeError(
            f"No narratable segments extracted for trial {trial_name!r}."
        )
    segments = apply_llm_narration_to_segments(
        segments,
        backend=narration_backend,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        hf_model=hf_narration_model,
        hf_token=hf_token,
        timeout_sec=narration_llm_timeout_sec,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{trial_name}_narration.wav"

    converter = tts_converter or BarkTTSConverter(
        voice_preset=voice_preset, device=device
    )
    converter.build_trial_audio_track(
        segments=segments,
        output_audio_path=wav_path,
        min_segment_seconds=min_segment_seconds,
        or_ambience=or_ambience,
    )
    return str(wav_path)


# ---------------------------------------------------------------------------
# moviepy-based muxing / comparison
# ---------------------------------------------------------------------------


def _import_moviepy():
    """Import moviepy lazily so pure-unit tests don't need the dep."""
    try:
        from moviepy import (  # type: ignore[import-not-found]
            AudioFileClip,
            CompositeVideoClip,
            ImageSequenceClip,
            TextClip,
            VideoFileClip,
            clips_array,
        )
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "moviepy>=2.0 is required for the compare pipeline. "
            "Install it with `pip install 'moviepy>=2.0.0'`."
        ) from e
    return {
        "AudioFileClip": AudioFileClip,
        "CompositeVideoClip": CompositeVideoClip,
        "ImageSequenceClip": ImageSequenceClip,
        "TextClip": TextClip,
        "VideoFileClip": VideoFileClip,
        "clips_array": clips_array,
    }


def _load_generated_video(path: Path, fps_hint: Optional[float] = None):
    """Build a moviepy clip from either an MP4 file or a dir of PNGs."""
    mp = _import_moviepy()
    if path.is_dir():
        frames = sorted(
            list(path.glob("*.png"))
            + list(path.glob("*.jpg"))
            + list(path.glob("*.jpeg"))
        )
        if not frames:
            raise RuntimeError(
                f"No image frames found under {path}. Expected .png/.jpg files."
            )
        fps = float(fps_hint) if fps_hint else 5.0
        return mp["ImageSequenceClip"]([str(f) for f in frames], fps=fps)
    if not path.exists():
        raise FileNotFoundError(f"Generated video path does not exist: {path}")
    return mp["VideoFileClip"](str(path))


def mux_audio_to_raw_video(
    trial_name: str,
    task: str,
    data_root: str | Path,
    audio_path: str | Path,
    output_dir: str | Path,
) -> str:
    """Mux the shared narration onto the original JIGSAWS ``.avi``.

    Writes ``<output_dir>/<trial_name>_raw_narrated.mp4`` and returns
    that path.  The AVI is H.264 re-encoded on the way out because
    browsers (and moviepy's in-notebook preview) do not play the raw
    MJPEG that JIGSAWS ships.
    """
    mp = _import_moviepy()
    assets = _locate_trial_assets(trial_name, task, data_root)
    if not assets["video"].exists():
        raise FileNotFoundError(
            f"Raw video not found for {trial_name!r} at {assets['video']}"
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trial_name}_raw_narrated.mp4"

    video = mp["VideoFileClip"](str(assets["video"]))
    audio = mp["AudioFileClip"](str(audio_path))
    try:
        dur = min(float(video.duration), float(audio.duration))
        video_clip = video.subclipped(0, dur) if hasattr(video, "subclipped") else video.subclip(0, dur)
        audio_clip = audio.subclipped(0, dur) if hasattr(audio, "subclipped") else audio.subclip(0, dur)
        final = (
            video_clip.with_audio(audio_clip)
            if hasattr(video_clip, "with_audio")
            else video_clip.set_audio(audio_clip)
        )
        final.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            threads=2,
            logger=None,
        )
    finally:
        video.close()
        audio.close()
    return str(out_path)


def mux_audio_to_generated_video(
    generated_frames_dir: str | Path,
    audio_path: str | Path,
    trial_name: str,
    output_dir: str | Path,
    fps_hint: Optional[float] = None,
) -> str:
    """Mux the shared narration onto the SD-generated video.

    ``generated_frames_dir`` may be either a directory of PNG/JPG frames
    (rendered by the SD pipeline) or an already-assembled MP4 file.
    Writes ``<output_dir>/<trial_name>_generated_narrated.mp4``.
    """
    mp = _import_moviepy()
    gen_path = Path(generated_frames_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trial_name}_generated_narrated.mp4"

    video = _load_generated_video(gen_path, fps_hint=fps_hint)
    audio = mp["AudioFileClip"](str(audio_path))
    try:
        dur = min(float(video.duration), float(audio.duration))
        video_clip = video.subclipped(0, dur) if hasattr(video, "subclipped") else video.subclip(0, dur)
        audio_clip = audio.subclipped(0, dur) if hasattr(audio, "subclipped") else audio.subclip(0, dur)
        final = (
            video_clip.with_audio(audio_clip)
            if hasattr(video_clip, "with_audio")
            else video_clip.set_audio(audio_clip)
        )
        final.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            threads=2,
            logger=None,
        )
    finally:
        video.close()
        audio.close()
    return str(out_path)


def _make_label(
    TextClip, text: str, video_w: int, video_h: int, duration: float
):
    """Build a bottom-left text overlay that survives moviepy 1.x / 2.x API drift."""
    font_size = max(18, video_h // 14)
    try:
        clip = TextClip(
            text=text,
            font_size=font_size,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="label",
        )
    except TypeError:
        # moviepy 1.x style
        clip = TextClip(
            txt=text,
            fontsize=font_size,
            color="white",
            stroke_color="black",
            stroke_width=2,
        )
    # Position 12px from bottom-left.  moviepy 2.x uses .with_*; 1.x uses .set_*.
    pos = (12, video_h - font_size - 24)
    if hasattr(clip, "with_duration"):
        clip = clip.with_duration(duration).with_position(pos)
    else:
        clip = clip.set_duration(duration).set_position(pos)
    return clip


def side_by_side_comparison(
    raw_narrated_path: str | Path,
    generated_narrated_path: str | Path,
    output_path: str | Path,
) -> str:
    """Stack raw | generated side by side with text labels.

    Audio is taken from ``generated_narrated_path`` only (both inputs
    carry the same narration track, so this just avoids a redundant
    mix).  Returns the output path.
    """
    mp = _import_moviepy()
    raw_clip = mp["VideoFileClip"](str(raw_narrated_path))
    gen_clip = mp["VideoFileClip"](str(generated_narrated_path))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dur = min(float(raw_clip.duration), float(gen_clip.duration))
        raw_cut = raw_clip.subclipped(0, dur) if hasattr(raw_clip, "subclipped") else raw_clip.subclip(0, dur)
        gen_cut = gen_clip.subclipped(0, dur) if hasattr(gen_clip, "subclipped") else gen_clip.subclip(0, dur)
        # Resize so both panels share a height.
        target_h = min(raw_cut.h, gen_cut.h)
        if hasattr(raw_cut, "resized"):
            raw_cut = raw_cut.resized(height=target_h)
            gen_cut = gen_cut.resized(height=target_h)
        else:
            raw_cut = raw_cut.resize(height=target_h)
            gen_cut = gen_cut.resize(height=target_h)
        left_label = _make_label(
            mp["TextClip"], "Original", raw_cut.w, raw_cut.h, dur
        )
        right_label = _make_label(
            mp["TextClip"], "Generated", gen_cut.w, gen_cut.h, dur
        )
        raw_labeled = mp["CompositeVideoClip"]([raw_cut, left_label], size=(raw_cut.w, raw_cut.h))
        gen_labeled = mp["CompositeVideoClip"]([gen_cut, right_label], size=(gen_cut.w, gen_cut.h))
        side = mp["clips_array"]([[raw_labeled, gen_labeled]])
        # Audio: use the generated clip's track (identical to raw's).
        audio = gen_clip.audio
        if audio is not None:
            audio_cut = audio.subclipped(0, dur) if hasattr(audio, "subclipped") else audio.subclip(0, dur)
            side = (
                side.with_audio(audio_cut)
                if hasattr(side, "with_audio")
                else side.set_audio(audio_cut)
            )
        side.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            threads=2,
            logger=None,
        )
    finally:
        raw_clip.close()
        gen_clip.close()
    return str(out_path)


__all__ = [
    "generate_shared_audio_track",
    "mux_audio_to_raw_video",
    "mux_audio_to_generated_video",
    "side_by_side_comparison",
]
