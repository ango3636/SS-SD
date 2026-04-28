"""Streamlit app: pick a suturing trial, preview the real video, and
generate a real-vs-generated side-by-side comparison of configurable length.

Run with:

    streamlit run scripts/streamlit_compare.py

The app shells out to ``scripts/generate_eval_video.py`` (so the
generation path stays identical to the CLI), then embeds the resulting
MP4s back into the page.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
# ``suturing_pipeline`` (src) and script-local helpers when launched via `streamlit run`.
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from suturing_pipeline.data.data_utils import parse_metafile  # noqa: E402

from video_quality_metrics import compute_metrics_board  # noqa: E402


def _render_narration_downloads(run_dir: Path, key_prefix: str) -> None:
    """Show download buttons for narration transcript / segment JSON when present."""
    run_slug = run_dir.name
    artifacts: List[Tuple[Path, str, str, str]] = []
    tr = run_dir / "narration_transcript.txt"
    if tr.exists():
        artifacts.append(
            (
                tr,
                f"{run_slug}_narration_transcript.txt",
                "text/plain",
                "Narration transcript (timed)",
            )
        )
    sj = run_dir / "narration_segments.json"
    if sj.exists():
        artifacts.append(
            (
                sj,
                f"{run_slug}_narration_segments.json",
                "application/json",
                "Narration segments + kinematics (JSON)",
            )
        )
    for p in sorted(run_dir.glob("*_shared_narration_segments.json")):
        artifacts.append(
            (p, p.name, "application/json", f"Shared narration segments ({p.name})")
        )
    for p in sorted(run_dir.glob("*_shared_narration_transcript.txt")):
        artifacts.append(
            (p, p.name, "text/plain", f"Shared narration transcript ({p.name})")
        )
    if not artifacts:
        return
    with st.expander("Download narration exports", expanded=False):
        for i, (path, fname, mime, label) in enumerate(artifacts):
            st.download_button(
                label,
                data=path.read_bytes(),
                file_name=fname,
                mime=mime,
                key=f"{key_prefix}_narr_{path.name}_{i}",
            )


def _resolve_ci(base: Path, *parts: str) -> Path:
    """Join ``parts`` under ``base`` resolving each segment case-insensitively.

    JIGSAWS ships with capital-S ``Suturing/`` and mixed-case
    ``Experimental_setup/`` on disk, but older code here hard-coded
    lowercase.  macOS filesystems are case-insensitive so that worked
    locally; Linux (Colab) is case-sensitive and fails.  This resolver
    walks segment-by-segment and returns the first child whose name
    matches ignoring case.  If no real child matches, it falls back to
    the literal join so callers still get a path they can test with
    ``.exists()`` and surface in error messages.
    """
    cur = base
    for seg in parts:
        if not cur.is_dir():
            return cur.joinpath(*parts[parts.index(seg):])
        match: Optional[Path] = None
        try:
            for child in cur.iterdir():
                if child.name == seg:
                    match = child
                    break
                if match is None and child.name.lower() == seg.lower():
                    match = child
        except OSError:
            return cur.joinpath(*parts[parts.index(seg):])
        if match is None:
            return cur.joinpath(*parts[parts.index(seg):])
        cur = match
    return cur


DATA_ROOT = REPO_ROOT / "data" / "gdrive_cache"
VIDEO_DIR = _resolve_ci(DATA_ROOT, "suturing", "video")
# JIGSAWS meta: col1 trial, col2 self N/I/E, col3 GRS, cols4–9 GRS sub-scores
# (suturing_pipeline.data.jigsaws_metafile_layout). UI skill badge uses col 2.
META_FILE = _resolve_ci(DATA_ROOT, "suturing", "meta_file_Suturing.txt")
EXP_SETUP_ROOT = _resolve_ci(
    DATA_ROOT,
    "Experimental_setup",
    "suturing",
    "balanced",
    "gestureclassification",
    "onetrialout",
)
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "eval_video" / "streamlit_runs"
PREVIEW_CACHE = OUTPUT_ROOT / "_previews"
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_eval_video.py"

SKILL_LABELS = {"N": "Novice", "I": "Intermediate", "E": "Expert"}

# Conservative default for the first run on a machine we've never seen
# (roughly matches a 2023 M-series Mac running 12-step sampling at 256).
_FALLBACK_SPF_PER_STEP = 12.0  # seconds per frame per diffusion step at 256px

# Also a soft cap: even the slowest sampling shouldn't exceed this per frame.
_MAX_REASONABLE_SPF = 900.0


@st.cache_data(show_spinner=False)
def scan_trials() -> List[Dict[str, object]]:
    """List every suturing trial that has both ``capture1.avi`` on disk
    and kinematics available."""
    if not VIDEO_DIR.is_dir():
        return []
    skill_map = _load_skill_map()
    trials: List[Dict[str, object]] = []
    seen: set[str] = set()
    for avi in sorted(VIDEO_DIR.glob("Suturing_*_capture1.avi")):
        m = re.match(r"(Suturing_[A-Z]\d{3})_capture1\.avi$", avi.name)
        if not m:
            continue
        trial = m.group(1)
        if trial in seen:
            continue
        seen.add(trial)
        skill = skill_map.get(trial, "?")
        trials.append(
            {
                "trial": trial,
                "video_path": str(avi),
                "skill_code": skill,
                "skill_label": SKILL_LABELS.get(skill, skill),
            }
        )
    return trials


def _load_skill_map() -> Dict[str, str]:
    """Map ``Suturing_*`` trial → single-letter N/I/E from meta file **column 2**."""
    if not META_FILE.exists():
        return {}
    label_to_letter = {"Novice": "N", "Intermediate": "I", "Expert": "E"}
    full = parse_metafile(META_FILE)
    out: Dict[str, str] = {}
    for name, label in full.items():
        if not str(name).startswith("Suturing_"):
            continue
        out[str(name)] = label_to_letter.get(label, (label or "?")[:1].upper())
    return out


@st.cache_data(show_spinner=False)
def scan_checkpoints() -> List[Dict[str, str]]:
    """List every ``step_*.pt`` under ``checkpoints/`` that has an ``args.json``
    sibling (so we know it's a real training run).
    """
    out: List[Dict[str, str]] = []
    if not CHECKPOINTS_ROOT.is_dir():
        return out
    for run_dir in sorted(CHECKPOINTS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        args_path = run_dir / "args.json"
        if not args_path.exists():
            continue
        ckpts = sorted(
            run_dir.glob("step_*.pt"),
            key=lambda p: int(re.search(r"step_(\d+)\.pt", p.name).group(1))
            if re.search(r"step_(\d+)\.pt", p.name)
            else 0,
        )
        for ckpt in ckpts:
            out.append(
                {
                    "run": run_dir.name,
                    "label": f"{run_dir.name} / {ckpt.name}",
                    "path": str(ckpt),
                }
            )
    return out


@st.cache_data(show_spinner=False)
def pick_fold_for_trial(trial_name: str) -> Tuple[Optional[int], str]:
    """Find a ``{N}_Out`` fold whose Train.txt contains *trial_name*.

    Every trial is in Train.txt for every fold except the single fold
    where it is the held-out test trial, so this always succeeds on a
    healthy JIGSAWS OneTrialOut setup.  Returns ``(None, reason)`` if
    nothing works.
    """
    if not EXP_SETUP_ROOT.is_dir():
        return None, f"Experimental_setup root not found: {EXP_SETUP_ROOT}"
    fold_dirs = sorted(
        (p for p in EXP_SETUP_ROOT.iterdir() if p.is_dir() and p.name.endswith("_Out")),
        key=lambda p: int(p.name.split("_Out")[0]) if p.name.split("_Out")[0].isdigit() else 10**9,
    )
    for fold in fold_dirs:
        m = re.match(r"(\d+)_Out", fold.name)
        if not m:
            continue
        fold_n = int(m.group(1))
        train_txt = fold / "itr_1" / "Train.txt"
        if not train_txt.exists():
            continue
        try:
            content = train_txt.read_text()
        except OSError:
            continue
        if trial_name in content:
            return fold_n, f"using fold {fold_n}_Out (trial present in Train.txt)"
    return None, f"No fold contains {trial_name} in Train.txt."


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def fmt_duration(seconds: float) -> str:
    """Format seconds as '23s' / '4m 12s' / '1h 03m 45s'."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _collect_run_metadata() -> List[Dict[str, float]]:
    """Harvest per-frame timing from every completed run metadata.json."""
    records: List[Dict[str, float]] = []
    for meta_path in REPO_ROOT.glob("outputs/eval_video/**/metadata.json"):
        try:
            m = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        avg = m.get("avg_seconds_per_frame")
        frames = m.get("rendered_frames") or 0
        args = m.get("args") or {}
        steps = args.get("num_inference_steps")
        size = args.get("image_size")
        if not avg or not frames or not steps or not size:
            continue
        records.append(
            {
                "avg_spf": float(avg),
                "frames": int(frames),
                "steps": int(steps),
                "size": int(size),
                "mtime": meta_path.stat().st_mtime,
            }
        )
    records.sort(key=lambda r: r["mtime"], reverse=True)
    return records


@st.cache_data(show_spinner=False, ttl=30)
def estimate_seconds_per_frame(steps: int, image_size: int) -> Tuple[float, str]:
    """Estimate seconds-per-frame for the requested config from past runs.

    Model: ``spf ~= k * steps * (image_size / 256)^2``.  We fit ``k`` from
    the most recent runs' ``avg_seconds_per_frame`` and report the median
    to avoid skew from a single unusually slow warm-up run.
    """
    records = _collect_run_metadata()
    if not records:
        k = _FALLBACK_SPF_PER_STEP
        source = "default (no prior runs)"
    else:
        recent = records[:8]
        ks = [
            r["avg_spf"] / max(r["steps"], 1) / ((r["size"] / 256.0) ** 2)
            for r in recent
        ]
        ks.sort()
        k = ks[len(ks) // 2]
        source = f"median of {len(recent)} past run(s)"
    spf = k * max(1, int(steps)) * ((int(image_size) / 256.0) ** 2)
    spf = min(spf, _MAX_REASONABLE_SPF)
    return spf, source


@st.cache_data(show_spinner=False)
def probe_video(video_path: str, mtime_ns: int) -> Dict[str, float]:
    """Return ``{fps, frame_count, duration_s}`` for ``video_path``.

    ``mtime_ns`` is only used as a cache key so that Streamlit regenerates
    the entry when the underlying file is replaced.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "duration_s": 0.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        return {"fps": 0.0, "frame_count": 0, "duration_s": 0.0}
    return {
        "fps": fps,
        "frame_count": frames,
        "duration_s": frames / fps,
    }


def _resolve_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable: system PATH first, then the bundled
    ``imageio-ffmpeg`` binary.
    """
    from shutil import which

    sys_ff = which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# Bump this if the transcoding pipeline changes so old cached files are
# ignored (earlier version wrote FMP4 which browsers can't play).
_PREVIEW_CACHE_VERSION = "v2-h264"


@st.cache_data(show_spinner=False)
def transcode_generated_to_h264(
    src_path_str: str,
    mtime_ns: int,
    keep_audio: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Re-encode a generated MP4 (written by OpenCV with the ``mp4v``
    fourcc = MPEG-4 Part 2) into browser-friendly H.264 and cache it.

    ``st.video`` / Chrome / Safari refuse to play ``mp4v``, so every
    generated clip needs this pass before display.  The encode is fast
    (a few seconds for short clips) and the result is cached keyed on
    the source mtime, so subsequent reruns are instant.

    Returns ``(h264_path, error_message)``. Exactly one will be None.
    """
    src_path = Path(src_path_str)
    if not src_path.exists() or src_path.stat().st_size < 1024:
        return None, f"Source missing or empty: {src_path}"

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        return None, (
            "No ffmpeg binary found. Install system ffmpeg (`brew install "
            "ffmpeg`) or `pip install imageio-ffmpeg`."
        )

    PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
    key = (
        f"gen_{slugify(src_path.parent.name)}_{slugify(src_path.stem)}"
        f"_{mtime_ns}_{_PREVIEW_CACHE_VERSION}_{'audio' if keep_audio else 'mute'}.mp4"
    )
    out_path = PREVIEW_CACHE / key
    if out_path.exists() and out_path.stat().st_size > 1024:
        return str(out_path), None

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        "-movflags", "+faststart",
    ]
    if not keep_audio:
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    cmd.append(str(out_path))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as e:
        return None, f"ffmpeg invocation failed: {e}"

    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1024:
        out_path.unlink(missing_ok=True)
        stderr = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = "\n".join(stderr[-8:]) if stderr else "(no stderr)"
        return None, f"ffmpeg transcode failed (rc={proc.returncode}):\n{tail}"
    return str(out_path), None


@st.cache_data(show_spinner=False)
def transcode_for_preview(
    avi_path_str: str,
    mtime_ns: int,
    start_seconds: float,
    max_seconds: int,
    target_height: int,
) -> Tuple[Optional[str], Optional[str]]:
    """Transcode ``avi_path`` to a browser-friendly H.264 MP4 and cache it.

    Browsers (and therefore ``st.video``) can't play raw AVI files, and
    the ``mp4v``/``FMP4`` fourcc that OpenCV writes is MPEG-4 Part 2
    which Chrome/Safari also refuse.  We shell out to ``ffmpeg`` (system
    or the ``imageio-ffmpeg`` bundled binary) to produce H.264 + AAC-less
    MP4 with ``+faststart`` so it streams immediately.

    Returns ``(mp4_path, error_message)``. Exactly one will be None.
    """
    avi_path = Path(avi_path_str)
    if not avi_path.exists():
        return None, f"Source video missing: {avi_path}"

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        return None, (
            "No ffmpeg binary found. Install system ffmpeg (`brew install "
            "ffmpeg`) or `pip install imageio-ffmpeg`."
        )

    PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
    start_tag = f"{max(0.0, float(start_seconds)):.2f}".replace(".", "p")
    key = (
        f"{slugify(avi_path.stem)}_{mtime_ns}_h{target_height}"
        f"_t{start_tag}_s{max_seconds}_{_PREVIEW_CACHE_VERSION}.mp4"
    )
    out_path = PREVIEW_CACHE / key
    if out_path.exists() and out_path.stat().st_size > 1024:
        return str(out_path), None

    # scale=-2:H keeps aspect and forces even dimensions (H.264 needs it).
    vf = f"scale=-2:{int(target_height)}"
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    # Put -ss BEFORE -i for fast seek (keyframe-accurate enough for a preview).
    if start_seconds > 0:
        cmd.extend(["-ss", f"{float(start_seconds):.3f}"])
    cmd.extend(["-i", str(avi_path)])
    if max_seconds > 0:
        cmd.extend(["-t", str(int(max_seconds))])
    cmd.extend([
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        str(out_path),
    ])

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as e:
        return None, f"ffmpeg invocation failed: {e}"

    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1024:
        out_path.unlink(missing_ok=True)
        stderr = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = "\n".join(stderr[-8:]) if stderr else "(no stderr)"
        return None, f"ffmpeg failed (rc={proc.returncode}):\n{tail}"
    return str(out_path), None


# Matches generator log lines like:
#   "  [3/50] src_f=12 gesture=G1 (gid=1, ...) anchor=... seed=0 (187.2s)"
_PROGRESS_RE = re.compile(
    r"\[(?P<i>\d+)/(?P<n>\d+)\].*\((?P<dt>\d+(?:\.\d+)?)s\)\s*$"
)


def run_generation(
    checkpoint_path: str,
    trial_name: str,
    capture: int,
    duration_seconds: float,
    fps_out: float,
    start_frame: int,
    num_inference_steps: int,
    image_size: int,
    anchor_mode: str,
    init_strength: float,
    held_out: Optional[int],
    run_name: str,
    log_container,
    expected_frames: int,
    initial_spf_est: float,
    progress_bar,
    eta_text,
    enable_narration: bool,
    tts_provider: str,
    tts_voice: str,
    narrate_sidebyside: bool,
    narration_default_outputs: bool,
    narration_backend: str = "template",
    ollama_base_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.2",
    hf_narration_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    hf_token: str = "",
    foley_dir: str = "",
    foley_gain_db: float = -12.0,
    foley_align: str = "start",
) -> Tuple[int, Path]:
    """Invoke ``scripts/generate_eval_video.py`` and stream its stdout into
    the given streamlit container, updating ``progress_bar`` / ``eta_text``
    with a rolling ETA as frames complete.  Returns ``(returncode, output_dir)``.
    """
    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()

    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--checkpoint", checkpoint_path,
        "--data_root", str(DATA_ROOT),
        "--output_dir", str(OUTPUT_ROOT),
        "--run_name", run_name,
        "--trial_name", trial_name,
        "--dataset_split", "train",
        "--capture", str(capture),
        "--duration_seconds", f"{duration_seconds:g}",
        "--fps_out", f"{fps_out:g}",
        "--frame_step", "6",
        "--start_frame", str(start_frame),
        "--num_inference_steps", str(num_inference_steps),
        "--image_size", str(image_size),
        "--anchor_mode", anchor_mode,
        "--init_strength", f"{init_strength:g}",
    ]
    if held_out is not None:
        cmd.extend(["--held_out", str(held_out)])
    if enable_narration:
        cmd.extend(
            [
                "--enable_narration",
                "--tts_provider",
                tts_provider,
                "--tts_voice",
                tts_voice,
            ]
        )
        if narrate_sidebyside:
            cmd.append("--narrate_sidebyside")
        if narration_default_outputs:
            cmd.append("--narration_default_outputs")
        nb = (narration_backend or "template").strip().lower()
        if nb != "template":
            cmd.extend(["--narration_backend", nb])
            if nb == "ollama":
                cmd.extend(
                    [
                        "--ollama_base_url",
                        ollama_base_url,
                        "--ollama_model",
                        ollama_model,
                    ]
                )
            if nb in ("huggingface", "hf"):
                cmd.extend(["--hf_narration_model", hf_narration_model])
                tok = (hf_token or "").strip()
                if tok:
                    env["HF_TOKEN"] = tok
        fd = (foley_dir or "").strip()
        if fd:
            cmd.extend(
                [
                    "--foley_dir",
                    fd,
                    "--foley_gain_db",
                    str(float(foley_gain_db)),
                    "--foley_align",
                    str(foley_align) if foley_align in ("start", "center") else "start",
                ]
            )

    log_container.code(" ".join(cmd), language="bash")
    live = log_container.empty()
    buf: List[str] = []

    t_start = time.time()
    per_frame_times: List[float] = []

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    # Initial ETA before any frames have landed.
    progress_bar.progress(0.0, text="starting generator...")
    eta_text.info(
        f"Waiting for first frame... Initial estimate: "
        f"~{fmt_duration(expected_frames * initial_spf_est)} "
        f"({initial_spf_est:.1f}s/frame * {expected_frames} frames)"
    )

    for line in proc.stdout:
        stripped = line.rstrip("\n")
        buf.append(stripped)
        live.code("\n".join(buf[-400:]))

        m = _PROGRESS_RE.search(stripped)
        if m:
            i = int(m["i"])
            n = int(m["n"]) or max(1, expected_frames)
            dt = float(m["dt"])
            per_frame_times.append(dt)

            # Rolling mean of last 5 frames for ETA.
            window = per_frame_times[-5:]
            est_spf = sum(window) / len(window)
            remaining = max(0, n - i)
            eta_s = remaining * est_spf
            elapsed = time.time() - t_start

            frac = min(1.0, max(0.0, i / n))
            progress_bar.progress(
                frac,
                text=f"frame {i}/{n} ({frac * 100:.0f}%)",
            )
            eta_text.info(
                f"**{i}/{n} frames** rendered · last frame {dt:.1f}s · "
                f"avg {est_spf:.1f}s/frame  \n"
                f"Elapsed: {fmt_duration(elapsed)} · "
                f"ETA: **{fmt_duration(eta_s)}** "
                f"(~{fmt_duration(elapsed + eta_s)} total)"
            )

    proc.wait()
    progress_bar.progress(1.0, text="done")
    return proc.returncode, out_dir


def render_video(path: Path, caption: str, keep_audio: bool = False) -> None:
    if not path.exists() or path.stat().st_size < 1024:
        st.warning(f"{caption}: not produced (`{path.name}` missing or tiny).")
        return
    st.caption(caption)
    h264_path, err = transcode_generated_to_h264(
        str(path),
        path.stat().st_mtime_ns,
        keep_audio=keep_audio,
    )
    if h264_path is None:
        st.warning(
            f"Couldn't transcode `{path.name}` for in-browser preview "
            f"({err}). The original file is on disk — use the download "
            "button below to view it in a local player."
        )
        with open(path, "rb") as f:
            st.download_button(
                f"Download {path.name}",
                data=f.read(),
                file_name=path.name,
                mime="video/mp4",
                key=f"dl_raw_{path}",
            )
        return
    st.video(h264_path)


@st.cache_data(show_spinner=False, ttl=15)
def scan_existing_runs() -> List[Dict[str, object]]:
    """List every run directory under ``OUTPUT_ROOT`` that has at least
    one playable artefact, newest first.

    Used by the "Load previous run" selector so a user who reruns
    Streamlit can revisit an already-generated clip without paying the
    multi-hour render cost again.
    """
    out: List[Dict[str, object]] = []
    if not OUTPUT_ROOT.is_dir():
        return out
    for run_dir in OUTPUT_ROOT.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        side = run_dir / "sidebyside.mp4"
        side_narrated = run_dir / "sidebyside_narrated.mp4"
        gen = run_dir / "generated.mp4"
        gen_narrated = run_dir / "generated_narrated.mp4"
        real = run_dir / "real.mp4"
        meta = run_dir / "metadata.json"
        # Playable = file exists and isn't the ~44-byte zero-frame husk.
        has_any = any(
            p.exists() and p.stat().st_size > 1024
            for p in (side, side_narrated, gen, gen_narrated, real)
        )
        if not has_any:
            continue
        out.append(
            {
                "name": run_dir.name,
                "dir": str(run_dir),
                "mtime": run_dir.stat().st_mtime,
                "has_side": side.exists() and side.stat().st_size > 1024,
                "has_side_narrated": side_narrated.exists()
                and side_narrated.stat().st_size > 1024,
                "has_gen": gen.exists() and gen.stat().st_size > 1024,
                "has_gen_narrated": gen_narrated.exists()
                and gen_narrated.stat().st_size > 1024,
                "has_real": real.exists() and real.stat().st_size > 1024,
                "has_meta": meta.exists(),
            }
        )
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def render_previous_run(run_info: Dict[str, object], fps_hint: float) -> None:
    """Render all artefacts of an already-completed run (video previews,
    metrics board, metadata).  Used to let the user resurrect a run
    after a Streamlit restart without regenerating anything.
    """
    run_dir = Path(str(run_info["dir"]))
    side_path = run_dir / "sidebyside.mp4"
    side_narrated_path = run_dir / "sidebyside_narrated.mp4"
    gen_path = run_dir / "generated.mp4"
    gen_narrated_path = run_dir / "generated_narrated.mp4"
    real_path = run_dir / "real.mp4"
    meta_path = run_dir / "metadata.json"

    st.caption(
        f"Run directory: `{run_dir.relative_to(REPO_ROOT)}`  \n"
        f"Last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(run_info['mtime'])))}"
    )

    selected_side = side_narrated_path if side_narrated_path.exists() else side_path
    selected_gen = gen_narrated_path if gen_narrated_path.exists() else gen_path
    meta = None
    narration_default_outputs = False
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = None
        if isinstance(meta, dict):
            narration_default_outputs = bool(
                ((meta.get("narration") or {}).get("default_outputs_narrated"))
            )
    side_has_audio = selected_side == side_narrated_path or (
        narration_default_outputs and selected_side == side_path
    )
    gen_has_audio = selected_gen == gen_narrated_path or (
        narration_default_outputs and selected_gen == gen_path
    )

    render_video(
        selected_side,
        "Side-by-side (REAL | GEN + narration)"
        if side_has_audio
        else "Side-by-side (REAL | GEN)",
        keep_audio=side_has_audio,
    )

    sub_a, sub_b = st.columns(2)
    with sub_a:
        render_video(real_path, "Real (resampled)")
    with sub_b:
        render_video(
            selected_gen,
            "Generated + narration" if gen_has_audio else "Generated",
            keep_audio=gen_has_audio,
        )

    if real_path.exists() and gen_path.exists():
        try:
            render_metrics_board(real_path, gen_path, fps_hint=float(fps_hint))
        except Exception as e:  # pragma: no cover - defensive UI path
            st.info(f"Metrics board unavailable: {e}")

    if meta:
        with st.expander("Run metadata", expanded=False):
            st.json(meta)

    if selected_side.exists() and selected_side.stat().st_size > 1024:
        with open(selected_side, "rb") as f:
            st.download_button(
                "Download side-by-side MP4",
                data=f.read(),
                file_name=f"{run_info['name']}_{selected_side.name}",
                mime="video/mp4",
                key=f"dl_side_prev_{run_info['name']}",
            )

    _render_narration_downloads(run_dir, key_prefix=f"prev_{run_info['name']}")


@st.cache_data(show_spinner=False)
def _cached_metrics_board(
    real_path: str,
    gen_path: str,
    real_mtime_ns: int,
    gen_mtime_ns: int,
    fps_hint: float,
) -> Optional[dict]:
    """Cache the metrics board keyed on file mtimes.

    Streamlit can't cache dataclass instances reliably (mutable, nested),
    so we serialise to a plain dict and reconstruct on the other side.
    """
    board = compute_metrics_board(
        Path(real_path), Path(gen_path), fps_hint=fps_hint
    )
    if board is None:
        return None
    return {
        "fps": board.fps,
        "n_frames": board.n_frames,
        "similarity_score": board.similarity_score,
        "stability_score": board.stability_score,
        "movement_score": board.movement_score,
        "overall_score": board.overall_score,
        "mean_ssim": board.mean_ssim,
        "mean_gen_delta": board.mean_gen_delta,
        "mean_real_delta": board.mean_real_delta,
        "motion_correlation": board.motion_correlation,
        "timeline": board.timeline.to_dict(orient="list"),
        "instabilities": [vars(i) for i in board.instabilities],
        "explanation": board.explanation,
    }


def _score_color(score: float) -> str:
    if score >= 70:
        return "normal"
    if score >= 50:
        return "off"
    return "inverse"


def _fmt_mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60:d}:{s % 60:02d}"


def render_metrics_board(
    real_path: Path, gen_path: Path, fps_hint: float
) -> None:
    """Show a user-friendly metrics board comparing the two MP4s."""
    import pandas as pd  # local import so we only pay it after generation

    if not (real_path.exists() and gen_path.exists()):
        return

    with st.spinner("Scoring the generated clip..."):
        try:
            data = _cached_metrics_board(
                str(real_path),
                str(gen_path),
                real_path.stat().st_mtime_ns,
                gen_path.stat().st_mtime_ns,
                float(fps_hint),
            )
        except Exception as e:  # pragma: no cover - defensive UI path
            st.warning(f"Could not compute metrics: {e}")
            return

    if data is None:
        st.info(
            "Metrics board unavailable (couldn't read the rendered clips)."
        )
        return

    st.divider()
    st.subheader("Metrics board: how good is this generation?")
    st.caption(
        "Plain-language scores for this clip.  All three sub-scores are on "
        "the same 0-100 scale; higher is better."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Overall",
        f"{data['overall_score']:.0f} / 100",
        help=(
            "Equal-weighted mean of similarity, stability, and movement "
            "fidelity."
        ),
    )
    k2.metric(
        "Similarity to real",
        f"{data['similarity_score']:.0f} / 100",
        delta=f"SSIM {data['mean_ssim']:.2f}",
        delta_color="off",
        help=(
            "Per-frame SSIM between the generated frame and the real frame "
            "at the same timestep, averaged over the clip.  SSIM close to "
            "1.0 means the two images look the same; anything above ~0.6 "
            "means the same scene, same tools, same hand locations."
        ),
    )
    k3.metric(
        "Temporal stability",
        f"{data['stability_score']:.0f} / 100",
        delta=f"gen delta {data['mean_gen_delta']:.1f} vs real {data['mean_real_delta']:.1f}",
        delta_color="off",
        help=(
            "How close the generated clip's frame-to-frame pixel change is "
            "to the real clip's.  If the generated delta is much larger "
            "than real, the model is flickering or hallucinating motion -- "
            "both are instability."
        ),
    )
    k4.metric(
        "Movement fidelity",
        f"{data['movement_score']:.0f} / 100",
        delta=f"flow corr r = {data['motion_correlation']:+.2f}",
        delta_color="off",
        help=(
            "Pearson correlation of optical-flow magnitude between real "
            "and generated over the clip.  r close to +1 means the "
            "generated clip moves at the same moments and intensities as "
            "the real one -- the hallmark of 'optimal' expert movement."
        ),
    )

    with st.expander(
        "What these scores mean (and how they're computed)", expanded=False
    ):
        st.markdown(data["explanation"])
        st.markdown(
            "**Signals under the hood**  \n"
            "- _Similarity_ uses the **Structural Similarity Index (SSIM)** "
            "on greyscale 144-px copies, computed frame-by-frame.  SSIM "
            "is sensitive to structural content (edges, textures) rather "
            "than raw pixel error.  \n"
            "- _Temporal stability_ uses **mean absolute pixel delta "
            "between consecutive frames** in each clip.  We compare "
            "generated-delta to real-delta because the real clip gives "
            "us a data-driven baseline for \"how much a suturing scene is "
            "supposed to change per frame\".  Excess change = instability.  \n"
            "- _Movement fidelity_ uses **dense optical flow (Farneback)** "
            "on each clip and correlates the flow-magnitude time series.  "
            "High correlation means the model's motion profile tracks the "
            "real operator's motion profile -- the clearest quantitative "
            "proxy for 'the model is reproducing the expert's movement'."
        )

    timeline = pd.DataFrame(data["timeline"])
    timeline = timeline.set_index("time_s")

    st.markdown("**Per-frame similarity over time**")
    st.caption(
        "Dips here correspond to moments the generated frame looks least "
        "like the real frame at the same instant.  Hover to read the time."
    )
    st.line_chart(timeline[["ssim"]], height=180)

    st.markdown("**Motion profile: real vs generated**")
    st.caption(
        "Ideal expert movement shows smooth, correlated rises and falls in "
        "both curves.  A generated curve that spikes alone -- or stays "
        "flat while the real curve moves -- is a sign of unoptimal motion."
    )
    st.line_chart(timeline[["real_motion", "gen_motion"]], height=220)

    st.markdown("**Frame-to-frame change (stability signal)**")
    st.caption(
        "Generated should roughly hug the real line.  Every time the "
        "orange line jumps well above the blue one, the model is "
        "introducing change the real scene did not have."
    )
    st.line_chart(timeline[["real_delta", "gen_delta"]], height=200)

    instabilities = data["instabilities"]
    st.markdown("**Flagged instabilities**")
    if not instabilities:
        st.success(
            "No instabilities flagged in this clip -- the generated motion "
            "stayed within a normal multiple of the real clip's own "
            "frame-to-frame change."
        )
        return

    st.caption(
        "Each row marks a stretch where the generated clip changed far "
        "more than the real clip did at the same moment.  These are the "
        "timestamps to scrub to in the videos above."
    )
    rows = []
    for inst in instabilities:
        rows.append(
            {
                "Window": f"{_fmt_mmss(inst['start_s'])} - {_fmt_mmss(inst['end_s'])}",
                "Duration (s)": round(
                    max(inst["end_s"] - inst["start_s"], 0.0), 2
                ),
                "Severity": inst["severity"].upper(),
                "Excess vs real (peak)": f"+{inst['peak_excess_pct']:.0f}%",
                "Why it was flagged": inst["reason"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Suturing Real-vs-Generated",
        layout="wide",
        page_icon=":scissors:",
    )
    st.title("Suturing: real vs generated comparison")
    st.caption(
        "Pick a JIGSAWS suturing trial, preview the raw video, then render "
        "a side-by-side comparison with the trained diffusion model."
    )

    trials = scan_trials()
    checkpoints = scan_checkpoints()

    if not trials:
        st.error(
            f"No suturing videos found under `{VIDEO_DIR}`.\n\n"
            f"- DATA_ROOT: `{DATA_ROOT}` (exists={DATA_ROOT.is_dir()})\n"
            f"- VIDEO_DIR: `{VIDEO_DIR}` (exists={VIDEO_DIR.is_dir()})\n"
            f"- META_FILE: `{META_FILE}` (exists={META_FILE.exists()})\n"
            f"- EXP_SETUP_ROOT: `{EXP_SETUP_ROOT}` (exists={EXP_SETUP_ROOT.is_dir()})\n\n"
            "Make sure the JIGSAWS tree (with `Suturing/` and "
            "`Experimental_setup/`) is symlinked into "
            "`<repo>/data/gdrive_cache/`. Path segments are resolved "
            "case-insensitively, so `Suturing` vs `suturing` is fine."
        )
        return
    if not checkpoints:
        st.error(
            f"No trained checkpoints found under `{CHECKPOINTS_ROOT}`. "
            "Train an SD run first (see `scripts/train_sd.py`)."
        )
        return

    with st.sidebar:
        st.header("Trial")
        trial_labels = [
            f"{t['trial']}  ({t['skill_label']})" for t in trials
        ]
        trial_default = next(
            (i for i, t in enumerate(trials) if t["skill_code"] == "E"),
            0,
        )
        trial_idx = st.selectbox(
            "Suturing trial",
            options=list(range(len(trials))),
            index=trial_default,
            format_func=lambda i: trial_labels[i],
        )
        trial = trials[trial_idx]

        capture = st.radio(
            "Camera",
            options=[1, 2],
            horizontal=True,
            help="JIGSAWS provides two camera views per trial.",
        )

        st.header("Model")
        default_ckpt_idx = len(checkpoints) - 1
        for i, c in enumerate(checkpoints):
            if "50ep" in c["run"]:
                default_ckpt_idx = i
        ckpt_idx = st.selectbox(
            "Checkpoint",
            options=list(range(len(checkpoints))),
            index=default_ckpt_idx,
            format_func=lambda i: checkpoints[i]["label"],
        )
        checkpoint = checkpoints[ckpt_idx]

        st.header("Output length")
        duration = st.slider(
            "Duration (seconds)",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="How long the generated MP4 should play back.",
        )

        with st.expander("Advanced settings", expanded=False):
            fps_out = st.slider(
                "Playback FPS", 2.0, 15.0, 5.0, 0.5, key="fps_out"
            )
            num_inference_steps = st.slider(
                "Diffusion steps", 10, 50, 25, 1,
                help="More steps = sharper but slower.",
            )
            image_size = st.select_slider(
                "Image size",
                options=[128, 192, 256, 320],
                value=256,
            )
            anchor_mode = st.selectbox(
                "Temporal anchor",
                options=["none", "prev_gen", "flow_warp", "prev_real"],
                index=2,
                help="Init each frame's latent from a reference (reduces flicker). "
                "flow_warp uses dense optical flow between real frames to move "
                "the previous generation into position before denoising.",
            )
            init_strength = st.slider(
                "Init strength", 0.3, 1.0, 0.7, 0.05,
                help="1.0 = pure noise (no anchor). Lower = stronger temporal coherence.",
            )
            enable_narration = st.checkbox(
                "Enable narration audio",
                value=False,
                help="Generate gesture-aligned spoken commentary and mux into output MP4s.",
            )
            tts_provider = st.selectbox(
                "Narration provider",
                options=["bark"],
                index=0,
                disabled=not enable_narration,
                help="Bark (suno/bark) is the only supported backend.",
            )
            tts_voice = st.text_input(
                "Bark voice preset",
                value="v2/en_speaker_9",
                disabled=not enable_narration,
                help=(
                    "Pass any Bark voice preset string; default is 'v2/en_speaker_9'; "
                    "sounds like a calm clinical instructor."
                ),
            )
            narrate_sidebyside = st.checkbox(
                "Narrate side-by-side output",
                value=True,
                disabled=not enable_narration,
            )
            narration_default_outputs = st.checkbox(
                "Narrate default outputs (overwrite generated.mp4 / sidebyside.mp4)",
                value=True,
                disabled=not enable_narration,
                help="If disabled, narrated copies are written as *_narrated.mp4 instead.",
            )
            narration_backend = st.selectbox(
                "Narration text (before Bark)",
                options=["template", "ollama", "huggingface"],
                index=0,
                disabled=not enable_narration,
                help=(
                    "template: kinematic sentences. ollama: free local LLM. "
                    "huggingface: HF Inference API (HF_TOKEN env or token field below)."
                ),
            )
            ollama_base_url = "http://127.0.0.1:11434"
            ollama_model = "llama3.2"
            hf_narration_model = "meta-llama/Llama-3.2-1B-Instruct"
            hf_token_ui = ""
            if enable_narration and narration_backend == "ollama":
                ollama_base_url = st.text_input(
                    "Ollama base URL",
                    value=ollama_base_url,
                    help="Default http://127.0.0.1:11434 — run `ollama serve` locally.",
                )
                ollama_model = st.text_input(
                    "Ollama model",
                    value=ollama_model,
                    help="Example: llama3.2, mistral, qwen2.5:7b",
                )
            if enable_narration and narration_backend == "huggingface":
                hf_narration_model = st.text_input(
                    "Hugging Face model id",
                    value=hf_narration_model,
                )
                hf_token_ui = st.text_input(
                    "Hugging Face read token",
                    type="password",
                    value="",
                    help=(
                        "Optional if the process already has HF_TOKEN set. "
                        "The token is passed via environment to the child process "
                        "(not shown in the logged shell command)."
                    ),
                )
            foley_dir_ui = st.text_input(
                "Optional foley WAV directory",
                value="",
                disabled=not enable_narration,
                help=(
                    "Folder with G1.wav … G15.wav (one optional clip per gesture). "
                    "Mixed under Bark per segment. Leave empty to disable."
                ),
            )
            foley_gain_db_ui = st.slider(
                "WAV foley level (dB vs segment stem)",
                -30.0,
                -3.0,
                -12.0,
                1.0,
                disabled=not enable_narration,
            )
            foley_align_ui = st.selectbox(
                "WAV foley alignment",
                options=["start", "center"],
                index=0,
                disabled=not enable_narration,
            )
            if (
                enable_narration
                and narration_backend == "ollama"
                and os.environ.get("COLAB_RELEASE_TAG")
            ):
                st.warning(
                    "**Colab:** Ollama uses `127.0.0.1` on *this* notebook VM, not your "
                    "home PC. A Cloudflare tunnel does not bridge Python HTTP to your "
                    "laptop. Prefer **huggingface** + HF token here, or install Ollama "
                    "inside Colab."
                )

    existing_runs = scan_existing_runs()
    if existing_runs:
        default_expanded = st.session_state.get(
            "_prev_run_expander_open", False
        )
        with st.expander(
            f"Load a previously generated run ({len(existing_runs)} available)",
            expanded=default_expanded,
        ):
            st.caption(
                "Pick any completed run in `outputs/eval_video/streamlit_runs/` "
                "to replay it here.  Nothing is regenerated — the videos on "
                "disk are just transcoded to H.264 for in-browser preview "
                "(takes seconds, cached afterwards)."
            )
            run_labels = [
                f"{r['name']}  "
                f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(float(r['mtime'])))})"
                for r in existing_runs
            ]
            prev_idx = st.selectbox(
                "Previous run",
                options=list(range(len(existing_runs))),
                index=0,
                format_func=lambda i: run_labels[i],
                key="_prev_run_idx",
            )
            if st.button(
                "Show this run",
                key="_prev_run_show",
                use_container_width=True,
            ):
                st.session_state["_prev_run_expander_open"] = True
                st.session_state["_prev_run_active"] = existing_runs[prev_idx]

        active_prev = st.session_state.get("_prev_run_active")
        if active_prev is not None:
            st.divider()
            st.subheader(f"Previously generated: {active_prev['name']}")
            # Use the current sidebar fps_out as a hint; it only affects the
            # metrics-board timeline axis, not the video decoding.
            render_previous_run(
                active_prev, fps_hint=float(st.session_state.get("fps_out", 5.0))
            )
            st.divider()

    col_real, col_gen = st.columns(2)

    real_video_path = Path(trial["video_path"])
    if capture == 2:
        alt = real_video_path.with_name(
            real_video_path.name.replace("capture1", "capture2")
        )
        if alt.exists():
            real_video_path = alt

    probe = probe_video(
        str(real_video_path), real_video_path.stat().st_mtime_ns
    )
    src_fps = probe["fps"]
    src_duration = probe["duration_s"]
    if src_fps <= 0 or src_duration <= 0:
        st.error(f"Could not probe video metadata for `{real_video_path}`.")
        return

    max_start_s = max(0.0, src_duration - float(duration))
    start_key = f"start_s::{trial['trial']}::cap{capture}"
    if start_key in st.session_state:
        st.session_state[start_key] = min(
            float(st.session_state[start_key]), max_start_s
        )

    with col_real:
        st.subheader("Real video")
        st.caption(
            f"**{trial['trial']}** · {trial['skill_label']} · "
            f"`{real_video_path.name}`  \n"
            f"Source: {src_duration:.1f}s @ {src_fps:.2f} fps "
            f"({probe['frame_count']:.0f} frames)"
        )

        start_s = st.slider(
            "Start time (seconds)",
            min_value=0.0,
            max_value=float(round(max_start_s, 1)),
            value=float(st.session_state.get(start_key, 0.0)),
            step=0.5,
            key=start_key,
            help="Drag to pick which segment of the trial to preview and "
            "render. The preview below shows exactly that segment.",
        )
        end_s = min(src_duration, start_s + float(duration))
        st.caption(
            f"Segment: **{start_s:.1f}s - {end_s:.1f}s** "
            f"({end_s - start_s:.1f}s of {src_duration:.1f}s)"
        )

        with st.spinner("Transcoding preview (H.264)..."):
            mp4_preview, preview_err = transcode_for_preview(
                str(real_video_path),
                real_video_path.stat().st_mtime_ns,
                float(start_s),
                int(duration),
                target_height=360,
            )
        if mp4_preview is None:
            st.error(
                "Could not transcode the AVI for preview.\n\n"
                f"Source: `{real_video_path}`\n\n{preview_err or ''}"
            )
        else:
            st.video(mp4_preview)

    start_frame = int(round(start_s * src_fps))

    expected_frames = max(1, int(duration * fps_out))
    spf_est, spf_source = estimate_seconds_per_frame(
        int(num_inference_steps), int(image_size)
    )
    total_est_s = expected_frames * spf_est

    with col_gen:
        st.subheader("Generated comparison")
        st.write(
            f"Model: `{checkpoint['label']}`  \n"
            f"Segment: **{start_s:.1f}s - {end_s:.1f}s**  \n"
            f"Duration: **{duration}s** at **{fps_out:g} fps**  \n"
            f"~{expected_frames} generated frames "
            f"× {num_inference_steps} diffusion steps  \n"
            f"Start frame: **{start_frame}**"
        )

        est_level = (
            "warning" if total_est_s > 5 * 60 else "info"
        )
        eta_msg = (
            f"Estimated generation time: **~{fmt_duration(total_est_s)}** "
            f"(~{spf_est:.1f}s/frame × {expected_frames} frames)  \n"
            f"_Based on {spf_source}; actual time depends on your CPU/GPU._"
        )
        if est_level == "warning":
            st.warning(eta_msg)
        else:
            st.info(eta_msg)

        go = st.button(
            f"Generate side-by-side (~{fmt_duration(total_est_s)})",
            type="primary",
            use_container_width=True,
        )

    # Render post-generation output (side-by-side video, metrics board,
    # downloads) at the top level so it spans the full page width instead of
    # being trapped inside `col_gen`'s half-width column.
    output_placeholder = st.container()

    if go:
        fold_n, fold_msg = pick_fold_for_trial(trial["trial"])
        if fold_n is None:
            st.error(fold_msg)
            st.stop()

        run_name = (
            f"{slugify(trial['trial'])}_cap{capture}_"
            f"{slugify(Path(checkpoint['path']).stem)}_"
            f"t{start_s:.1f}s_{duration}s_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ).replace(".", "p")

        st.info(
            f"Running generator... ({fold_msg}). "
            f"Estimated **~{fmt_duration(total_est_s)}** "
            f"(~{spf_est:.1f}s/frame * {expected_frames} frames)."
        )
        progress_bar = st.progress(0.0, text="preparing...")
        eta_text = st.empty()
        log_box = st.expander("Generator log", expanded=False)
        t0 = time.time()
        with st.spinner(f"Generating {duration}s comparison for {trial['trial']}..."):
            rc, out_dir = run_generation(
                checkpoint_path=checkpoint["path"],
                trial_name=trial["trial"],
                capture=capture,
                duration_seconds=float(duration),
                fps_out=float(fps_out),
                start_frame=int(start_frame),
                num_inference_steps=int(num_inference_steps),
                image_size=int(image_size),
                anchor_mode=anchor_mode,
                init_strength=float(init_strength),
                held_out=fold_n,
                run_name=run_name,
                log_container=log_box,
                expected_frames=expected_frames,
                initial_spf_est=spf_est,
                progress_bar=progress_bar,
                eta_text=eta_text,
                enable_narration=bool(enable_narration),
                tts_provider=str(tts_provider),
                tts_voice=str(tts_voice),
                narrate_sidebyside=bool(narrate_sidebyside),
                narration_default_outputs=bool(narration_default_outputs),
                narration_backend=str(narration_backend),
                ollama_base_url=str(ollama_base_url),
                ollama_model=str(ollama_model),
                hf_narration_model=str(hf_narration_model),
                hf_token=str(hf_token_ui),
                foley_dir=str(foley_dir_ui),
                foley_gain_db=float(foley_gain_db_ui),
                foley_align=str(foley_align_ui),
            )
        dt = time.time() - t0
        if rc != 0:
            st.error(
                f"Generator exited with code {rc} after {dt:.1f}s. "
                "See log above for details."
            )
            st.stop()

        st.success(f"Done in {dt:.1f}s. Artefacts in `{out_dir}`.")

        with output_placeholder:
            side_path = out_dir / "sidebyside.mp4"
            side_narrated_path = out_dir / "sidebyside_narrated.mp4"
            gen_path = out_dir / "generated.mp4"
            gen_narrated_path = out_dir / "generated_narrated.mp4"
            real_path = out_dir / "real.mp4"
            meta_path = out_dir / "metadata.json"

            meta = None
            narration_default_outputs_meta = False
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    meta = None
                if isinstance(meta, dict):
                    narration_default_outputs_meta = bool(
                        ((meta.get("narration") or {}).get("default_outputs_narrated"))
                    )
            selected_side = side_narrated_path if side_narrated_path.exists() else side_path
            selected_gen = gen_narrated_path if gen_narrated_path.exists() else gen_path
            side_has_audio = selected_side == side_narrated_path or (
                narration_default_outputs_meta and selected_side == side_path
            )
            gen_has_audio = selected_gen == gen_narrated_path or (
                narration_default_outputs_meta and selected_gen == gen_path
            )
            render_video(
                selected_side,
                "Side-by-side (REAL | GEN + narration)"
                if side_has_audio
                else "Side-by-side (REAL | GEN)",
                keep_audio=side_has_audio,
            )

            sub_a, sub_b = st.columns(2)
            with sub_a:
                render_video(real_path, "Real (resampled)")
            with sub_b:
                render_video(
                    selected_gen,
                    "Generated + narration" if gen_has_audio else "Generated",
                    keep_audio=gen_has_audio,
                )

            if real_path.exists() and gen_path.exists():
                render_metrics_board(real_path, gen_path, fps_hint=float(fps_out))

            if meta:
                with st.expander("Run metadata", expanded=False):
                    st.json(meta)

            with open(selected_side, "rb") as f:
                st.download_button(
                    "Download side-by-side MP4",
                    data=f.read(),
                    file_name=f"{run_name}_{selected_side.name}",
                    mime="video/mp4",
                )

            _render_narration_downloads(out_dir, key_prefix=f"gen_{run_name}")


if __name__ == "__main__":
    main()
