"""Live quality metrics for real-vs-generated suturing clips.

Given the two MP4 artefacts that ``scripts/generate_eval_video.py``
writes (``real.mp4`` and ``generated.mp4``, same resolution and frame
rate), this module computes a small, explainable "metrics board" that
answers three questions a non-expert user might have after watching the
side-by-side:

1. **Where were instabilities?** -- moments where the generated clip
   flickers or jumps in a way the real clip does not.
2. **How do we know these are instabilities?** -- each flagged window
   carries a plain-language reason tied to a measurable signal.
3. **How well does the generated clip match the real, in terms of
   showing optimal (smooth, economical) movement?** -- we compare
   per-frame similarity and the *temporal motion profile* of both
   clips using optical flow.

Design notes
------------
Frames from ``real.mp4`` and ``generated.mp4`` are paired by index --
``generate_eval_video.py`` resamples both streams to the same
``fps_out``, so frame ``t`` in each file corresponds to the same
moment in clip time.

The expensive ops (SSIM, Farneback optical flow) run on a downsampled
greyscale copy (default 144 px tall) so a 30-75 frame comparison
completes in well under a second on a laptop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim_fn


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Frames are downsampled to this height before SSIM/optical-flow so we
# stay interactive even on CPU.  Generated videos are typically 256 px
# already, so this is a modest reduction.
_ANALYSIS_HEIGHT = 144

# Map mean SSIM into a 0-100 "similarity score".  Diffusion outputs
# rarely exceed 0.85 on JIGSAWS (scene is dim, hands are small), so we
# anchor 0.30 -> 0 and 0.90 -> 100 to keep the score readable.
_SSIM_LOW = 0.30
_SSIM_HIGH = 0.90

# "Excess flicker" is the amount by which the generated frame-to-frame
# pixel delta exceeds the real one.  We normalise by the real clip's
# own delta (i.e. "how much more jittery than the source") and map
# through exp(-k * relative_excess).  k=2 means ~14% excess -> 75,
# ~70% excess -> 25.
_STABILITY_DECAY = 2.0

# Flag an instability at frame t when:
#   gen_delta[t] > median + FLAG_SIGMA * MAD     (outlier in its own clip)
#   AND gen_delta[t] > FLAG_EXCESS_RATIO * real_delta[t]  (not just scene motion)
_FLAG_SIGMA = 2.5
_FLAG_EXCESS_RATIO = 1.8
# Consecutive flagged frames within this many frames merge into one window.
_FLAG_MERGE_GAP = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Instability:
    start_s: float
    end_s: float
    peak_excess_pct: float  # % that gen flicker exceeded real at the peak
    severity: str  # "low" | "medium" | "high"
    reason: str


@dataclass
class MetricsBoard:
    fps: float
    n_frames: int
    similarity_score: float  # 0-100
    stability_score: float  # 0-100
    movement_score: float  # 0-100
    overall_score: float  # 0-100
    mean_ssim: float
    mean_gen_delta: float
    mean_real_delta: float
    motion_correlation: float  # Pearson r of optical-flow magnitudes
    timeline: pd.DataFrame = field(repr=False)
    instabilities: List[Instability] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Video IO + preprocessing
# ---------------------------------------------------------------------------


def _read_frames(path: Path, max_frames: int = 600) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _downsample(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= target_h:
        return frame
    new_w = max(1, int(round(w * (target_h / h))))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _prepare(frames: List[np.ndarray], target_h: int) -> List[np.ndarray]:
    """Downsample and convert to greyscale uint8."""
    out: List[np.ndarray] = []
    for f in frames:
        d = _downsample(f, target_h)
        out.append(cv2.cvtColor(d, cv2.COLOR_RGB2GRAY))
    return out


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _median_absolute_deviation(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    # scale MAD so it's comparable to a stdev under Gaussian noise
    return mad * 1.4826 if mad > 0 else float(np.std(x))


def _farneback(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def _flow_magnitude(flow: np.ndarray) -> float:
    return float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))


def _severity(peak_excess_pct: float) -> str:
    if peak_excess_pct >= 200:
        return "high"
    if peak_excess_pct >= 80:
        return "medium"
    return "low"


def _flag_instabilities(
    gen_delta: np.ndarray,
    real_delta: np.ndarray,
    fps: float,
) -> List[Instability]:
    if len(gen_delta) < 3:
        return []
    med = float(np.median(gen_delta))
    mad = _median_absolute_deviation(gen_delta)
    threshold_outlier = med + _FLAG_SIGMA * max(mad, 1e-6)

    flagged = np.zeros(len(gen_delta), dtype=bool)
    for t in range(1, len(gen_delta)):
        if gen_delta[t] <= threshold_outlier:
            continue
        real_ref = max(real_delta[t], 1e-6)
        if gen_delta[t] < _FLAG_EXCESS_RATIO * real_ref:
            continue
        flagged[t] = True

    # Merge adjacent flagged frames into windows.
    windows: List[Tuple[int, int]] = []
    i = 0
    while i < len(flagged):
        if not flagged[i]:
            i += 1
            continue
        j = i
        while (
            j + 1 < len(flagged)
            and (flagged[j + 1] or (j + 2 < len(flagged) and flagged[j + 2]))
            and (j - i) < 50
        ):
            j += 1
        windows.append((i, j))
        i = j + 1
        # respect merge gap
        _ = _FLAG_MERGE_GAP  # reserved for future tuning

    out: List[Instability] = []
    for a, b in windows:
        window = gen_delta[a : b + 1]
        ref = np.maximum(real_delta[a : b + 1], 1e-6)
        excess_pct = float(np.max((window - ref) / ref) * 100.0)
        sev = _severity(excess_pct)
        start_s = a / fps if fps > 0 else float(a)
        end_s = (b + 1) / fps if fps > 0 else float(b + 1)
        duration_s = max(end_s - start_s, 1.0 / max(fps, 1e-6))
        reason = (
            f"Generated frame-to-frame change was {excess_pct:.0f}% larger "
            f"than the real clip over {duration_s:.1f}s, i.e. the model "
            "moved pixels more than the real scene did -- usually flicker, "
            "hallucinated edges, or a sudden identity shift."
        )
        out.append(
            Instability(
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                peak_excess_pct=round(excess_pct, 1),
                severity=sev,
                reason=reason,
            )
        )
    return out


def compute_metrics_board(
    real_path: Path,
    gen_path: Path,
    fps_hint: Optional[float] = None,
) -> Optional[MetricsBoard]:
    """Read both MP4s, score similarity/stability/movement, flag instabilities.

    Returns None if the videos can't be read or have no overlapping frames.
    """
    real_frames, real_fps = _read_frames(Path(real_path))
    gen_frames, gen_fps = _read_frames(Path(gen_path))
    if not real_frames or not gen_frames:
        return None
    fps = fps_hint or gen_fps or real_fps or 5.0

    n = min(len(real_frames), len(gen_frames))
    if n < 3:
        return None
    real_frames = real_frames[:n]
    gen_frames = gen_frames[:n]

    real_g = _prepare(real_frames, _ANALYSIS_HEIGHT)
    gen_g = _prepare(gen_frames, _ANALYSIS_HEIGHT)

    # Align sizes in case the two streams were written at different dims.
    H = min(real_g[0].shape[0], gen_g[0].shape[0])
    W = min(real_g[0].shape[1], gen_g[0].shape[1])
    real_g = [f[:H, :W] for f in real_g]
    gen_g = [f[:H, :W] for f in gen_g]

    ssim_per_frame = np.zeros(n, dtype=np.float64)
    gen_delta = np.zeros(n, dtype=np.float64)
    real_delta = np.zeros(n, dtype=np.float64)
    real_flow_mag = np.zeros(n, dtype=np.float64)
    gen_flow_mag = np.zeros(n, dtype=np.float64)

    for t in range(n):
        ssim_per_frame[t] = float(
            ssim_fn(real_g[t], gen_g[t], data_range=255)
        )
        if t == 0:
            continue
        gen_delta[t] = float(
            np.mean(
                np.abs(gen_g[t].astype(np.float32) - gen_g[t - 1].astype(np.float32))
            )
        )
        real_delta[t] = float(
            np.mean(
                np.abs(real_g[t].astype(np.float32) - real_g[t - 1].astype(np.float32))
            )
        )
        try:
            real_flow_mag[t] = _flow_magnitude(_farneback(real_g[t - 1], real_g[t]))
            gen_flow_mag[t] = _flow_magnitude(_farneback(gen_g[t - 1], gen_g[t]))
        except cv2.error:
            real_flow_mag[t] = 0.0
            gen_flow_mag[t] = 0.0

    mean_ssim = float(np.mean(ssim_per_frame))
    similarity_score = float(
        np.clip(
            (mean_ssim - _SSIM_LOW) / (_SSIM_HIGH - _SSIM_LOW) * 100.0, 0.0, 100.0
        )
    )

    # Stability: how close gen frame-to-frame motion is to the real one.
    # Only look at frames with movement (t >= 1).
    valid = slice(1, n)
    ref_delta = np.maximum(real_delta[valid], 1e-3)
    relative_excess = np.maximum(gen_delta[valid] - real_delta[valid], 0.0) / ref_delta
    mean_rel_excess = float(np.mean(relative_excess))
    stability_score = float(
        100.0 * math.exp(-_STABILITY_DECAY * mean_rel_excess)
    )

    # Movement: correlation of optical-flow magnitude between the clips.
    motion_corr = _pearson(real_flow_mag[valid], gen_flow_mag[valid])
    movement_score = float(np.clip(50.0 + 50.0 * motion_corr, 0.0, 100.0))

    overall = float(np.mean([similarity_score, stability_score, movement_score]))

    time_s = np.arange(n, dtype=np.float64) / max(fps, 1e-6)
    timeline = pd.DataFrame(
        {
            "time_s": time_s,
            "ssim": ssim_per_frame,
            "gen_delta": gen_delta,
            "real_delta": real_delta,
            "gen_motion": gen_flow_mag,
            "real_motion": real_flow_mag,
        }
    )

    instabilities = _flag_instabilities(gen_delta, real_delta, fps)

    explanation = _build_explanation(
        similarity_score=similarity_score,
        stability_score=stability_score,
        movement_score=movement_score,
        motion_corr=motion_corr,
        mean_ssim=mean_ssim,
        instabilities=instabilities,
    )

    return MetricsBoard(
        fps=fps,
        n_frames=n,
        similarity_score=round(similarity_score, 1),
        stability_score=round(stability_score, 1),
        movement_score=round(movement_score, 1),
        overall_score=round(overall, 1),
        mean_ssim=round(mean_ssim, 3),
        mean_gen_delta=round(float(np.mean(gen_delta[valid])), 3),
        mean_real_delta=round(float(np.mean(real_delta[valid])), 3),
        motion_correlation=round(motion_corr, 3),
        timeline=timeline,
        instabilities=instabilities,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Plain-language interpretation
# ---------------------------------------------------------------------------


def _grade(score: float) -> str:
    if score >= 80:
        return "very good"
    if score >= 65:
        return "good"
    if score >= 50:
        return "fair"
    if score >= 35:
        return "weak"
    return "poor"


def _build_explanation(
    similarity_score: float,
    stability_score: float,
    movement_score: float,
    motion_corr: float,
    mean_ssim: float,
    instabilities: List[Instability],
) -> str:
    n_flag = len(instabilities)
    high_sev = sum(1 for i in instabilities if i.severity == "high")

    sim_verdict = _grade(similarity_score)
    stab_verdict = _grade(stability_score)
    move_verdict = _grade(movement_score)

    if motion_corr >= 0.6:
        move_note = (
            "the generated clip's motion rises and falls at roughly the same "
            "moments as the real clip, which is the hallmark of 'optimal' "
            "expert movement -- smooth, purposeful strokes timed to the task."
        )
    elif motion_corr >= 0.2:
        move_note = (
            "the generated clip moves somewhat in sync with the real one, but "
            "the timing of strokes drifts.  Expert suturing is characterised "
            "by tightly timed motion, so watch for stalls or bursts that the "
            "real clip doesn't have."
        )
    else:
        move_note = (
            "the generated clip's motion is mostly uncorrelated with the real "
            "clip.  That usually means hallucinated movement (the model is "
            "animating the scene independently) or frozen output -- neither "
            "is what an expert demonstration looks like."
        )

    if n_flag == 0:
        instab_note = (
            "No instability spikes were flagged: frame-to-frame changes in "
            "the generated clip stayed within a normal multiple of the real "
            "clip's own motion."
        )
    else:
        instab_note = (
            f"Flagged **{n_flag} instability window(s)** "
            f"({high_sev} high-severity).  Each one is a stretch where the "
            "generated pixels shifted far more than the real scene did, "
            "which we treat as flicker, identity drift, or hallucinated "
            "motion rather than real movement."
        )

    return (
        f"**Similarity ({_grade(similarity_score)})**: mean SSIM against the "
        f"real frame at the same timestep is {mean_ssim:.2f}. SSIM = 1.0 "
        f"means a perfect pixel-level copy; values above ~0.6 mean the "
        f"scene, tools and hand positions are recognisably the same.  \n"
        f"**Stability ({stab_verdict})**: compares frame-to-frame brightness "
        f"changes in the generated clip to the same changes in the real "
        f"clip.  The real clip is our baseline for 'how much a suturing "
        f"scene is supposed to change between frames'; the generated clip "
        f"should not change noticeably more than that.  \n"
        f"**Movement fidelity ({move_verdict}, r={motion_corr:+.2f})**: "
        f"{move_note}  \n"
        f"{instab_note}"
    )


def metrics_board_to_jsonable(board: MetricsBoard) -> dict:
    """Serialize scalar metrics + instability list for JSON export."""
    from dataclasses import asdict

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
        "instabilities": [asdict(i) for i in board.instabilities],
        "explanation": board.explanation,
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    _ap = argparse.ArgumentParser(description="Clip-pair metrics (SSIM, flow).")
    _ap.add_argument("--real", type=Path, required=True)
    _ap.add_argument("--gen", type=Path, required=True)
    _ap.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON.",
    )
    _cli = _ap.parse_args()
    _board = compute_metrics_board(_cli.real, _cli.gen)
    if _board is None:
        print(
            "Could not compute metrics (missing videos or too few frames).",
            file=sys.stderr,
        )
        sys.exit(1)
    _payload = metrics_board_to_jsonable(_board)
    if _cli.out_json is not None:
        _cli.out_json.parent.mkdir(parents=True, exist_ok=True)
        _cli.out_json.write_text(json.dumps(_payload, indent=2), encoding="utf-8")
        print(f"Wrote {_cli.out_json}")
    else:
        print(json.dumps(_payload, indent=2))
