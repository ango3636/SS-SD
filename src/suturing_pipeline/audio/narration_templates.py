from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from suturing_pipeline.data.data_utils import (
    filter_expert_trials,
    parse_kinematics,
    parse_transcription,
)

GESTURE_DESCRIPTIONS: Dict[str, str] = {
    "G1": "Reaching for needle with right hand",
    "G2": "Positioning needle",
    "G3": "Pushing needle through tissue",
    "G4": "Transferring needle from left to right",
    "G5": "Moving to center with needle in grip",
    "G6": "Pulling suture with left hand",
    "G7": "Pulling suture with right hand",
    "G8": "Orienting needle",
    "G9": "Using right hand to help tighten suture",
    "G10": "Loosening more suture",
    "G11": "Dropping suture and moving to end position",
    "G12": "Reaching for needle with left hand",
    "G13": "Making C-loop to the left",
    "G14": "Making C-loop to the right",
    "G15": "Reaching for suture with right hand",
}

# Text prompts for AudioGen OR ambience (one clip per narration segment, keyed by gesture).
GESTURE_AMBIENCE: Dict[str, str] = {
    "DEFAULT": (
        "Operating room background: quiet ventilation hum, distant sterile "
        "cart wheels, muffled monitor tones, no speech"
    ),
    "G1": (
        "Operating room ambience, surgeon reaching for needle driver, "
        "subtle metal instrument tray clicks, HVAC hum"
    ),
    "G2": (
        "OR room tone with focused needle positioning, light fabric rustle, "
        "steady monitor beep in background"
    ),
    "G3": (
        "Tissue needle pass in operating theatre, soft suction hiss, "
        "cautery standby hum, quiet staff movement"
    ),
    "G4": (
        "Needle handoff between surgeons in OR, latex glove sounds, "
        "distant suction and low ventilation"
    ),
    "G5": (
        "Central surgical field ambience, needle held steady, "
        "instrument trolley resonance, subdued OR chatter"
    ),
    "G6": (
        "Suture pull left hand in OR, thread tension through quiet room, "
        "monitor pulse ox rhythm faintly"
    ),
    "G7": (
        "Right-hand suture traction in operating room, subtle rope-on-glove sound, "
        "background suction idle"
    ),
    "G8": (
        "Needle orientation at operative site, small metal clicks, "
        "standard OR air handling noise"
    ),
    "G9": (
        "Knot tightening in OR, light cord rub, electrocautery unit fan, "
        "no voices in foreground"
    ),
    "G10": (
        "More suture paid out in sterile field, packaging rustle, "
        "quiet OR ambience with distant alarms muted"
    ),
    "G11": (
        "Suture dropped, instruments settling on mayo stand, "
        "end-of-motion OR background hum"
    ),
    "G12": (
        "Left-hand reach for needle in operating theatre, tray percussion, "
        "soft boot scuff on linoleum"
    ),
    "G13": (
        "C-loop motion left in OR, thread swish, low-frequency room rumble"
    ),
    "G14": (
        "C-loop motion right in OR, thread swish, electrosurgery cart idle tone"
    ),
    "G15": (
        "Reach for suture packet in OR, crinkle of sterile wrapper, "
        "ventilation and distant equipment beeps"
    ),
}

ABSOLUTE_SPEED_THRESHOLDS_MM_S = {
    "slow_max": 30.0,
    "fast_min": 80.0,
}

_EPS = 1e-8


@dataclass(frozen=True)
class SpeedStats:
    mean: float
    std: float
    count: int


def _resolve_ci(path: Path) -> Path:
    """Resolve path segments case-insensitively where possible."""
    parts = path.parts
    if not parts:
        return path
    resolved = Path(parts[0])
    for segment in parts[1:]:
        candidate = resolved / segment
        if candidate.exists():
            resolved = candidate
            continue
        seg_lower = segment.lower()
        try:
            matches = [p for p in resolved.iterdir() if p.name.lower() == seg_lower]
        except OSError:
            return path
        if not matches:
            return path
        resolved = matches[0]
    return resolved


def _speed_per_row(kinematic_segment: np.ndarray) -> np.ndarray:
    left = np.asarray(kinematic_segment[:, 0:3], dtype=np.float64)
    right = np.asarray(kinematic_segment[:, 38:41], dtype=np.float64)
    left_mag = np.linalg.norm(left, axis=1)
    right_mag = np.linalg.norm(right, axis=1)
    return 0.5 * (left_mag + right_mag)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _classify_speed(
    speed_value: float,
    gesture_label: Optional[str],
    expert_speed_stats: Optional[Dict[str, SpeedStats | Dict[str, float]]],
    min_count_for_empirical: int,
) -> tuple[str, str]:
    if expert_speed_stats and gesture_label:
        stats_raw = expert_speed_stats.get(gesture_label)
        if isinstance(stats_raw, SpeedStats):
            stats = stats_raw
        elif isinstance(stats_raw, dict):
            stats = SpeedStats(
                mean=float(stats_raw.get("mean", 0.0)),
                std=float(stats_raw.get("std", 0.0)),
                count=int(stats_raw.get("count", 0)),
            )
        else:
            stats = None
        if stats is not None and stats.count >= min_count_for_empirical:
            low = stats.mean - stats.std
            high = stats.mean + stats.std
            if speed_value < low:
                return "slow", "empirical"
            if speed_value > high:
                return "fast", "empirical"
            return "optimal", "empirical"

    slow_max = ABSOLUTE_SPEED_THRESHOLDS_MM_S["slow_max"]
    fast_min = ABSOLUTE_SPEED_THRESHOLDS_MM_S["fast_min"]
    if speed_value < slow_max:
        return "slow", "absolute_fallback"
    if speed_value > fast_min:
        return "fast", "absolute_fallback"
    return "optimal", "absolute_fallback"


def extract_kinematic_summary(
    kinematic_segment: np.ndarray,
    gesture_label: Optional[str] = None,
    expert_speed_stats: Optional[Dict[str, SpeedStats | Dict[str, float]]] = None,
    min_count_for_empirical: int = 8,
) -> Dict[str, float | str]:
    """Summarize a gesture segment from JIGSAWS kinematics (N, 76)."""
    seg = np.asarray(kinematic_segment, dtype=np.float64)
    if seg.ndim != 2 or seg.shape[1] < 76:
        raise ValueError(
            f"kinematic_segment must have shape (N, 76+), got {seg.shape}"
        )
    if seg.shape[0] == 0:
        return {
            "avg_velocity_left": 0.0,
            "avg_velocity_right": 0.0,
            "avg_gripper_left": 0.0,
            "avg_gripper_right": 0.0,
            "motion_smoothness": 0.0,
            "avg_speed_magnitude": 0.0,
            "speed_rating": "slow",
            "speed_rating_source": "absolute_fallback",
            "segment_length_frames": 0,
        }

    avg_velocity_left = _safe_mean(seg[:, 0:3])
    avg_velocity_right = _safe_mean(seg[:, 38:41])
    avg_gripper_left = _safe_mean(seg[:, 37])
    avg_gripper_right = _safe_mean(seg[:, 75])
    speed_rows = _speed_per_row(seg)
    speed_var = float(np.nanvar(speed_rows)) if speed_rows.size else 0.0
    motion_smoothness = float(1.0 / (speed_var + _EPS))
    avg_speed = _safe_mean(speed_rows)
    speed_rating, speed_source = _classify_speed(
        speed_value=avg_speed,
        gesture_label=gesture_label,
        expert_speed_stats=expert_speed_stats,
        min_count_for_empirical=min_count_for_empirical,
    )

    return {
        "avg_velocity_left": avg_velocity_left,
        "avg_velocity_right": avg_velocity_right,
        "avg_gripper_left": avg_gripper_left,
        "avg_gripper_right": avg_gripper_right,
        "motion_smoothness": motion_smoothness,
        "avg_speed_magnitude": avg_speed,
        "speed_rating": speed_rating,
        "speed_rating_source": speed_source,
        "segment_length_frames": int(seg.shape[0]),
    }


def max_narration_words_for_duration(duration_seconds: float) -> int:
    """Upper bound on narration length for LLM / TTS timing (word count)."""
    d = float(duration_seconds)
    if d < 2.0:
        return 8
    if d < 4.0:
        return 15
    return 25


def build_llm_prompt(
    gesture_label: str,
    gesture_description: str,
    kinematic_summary: Dict[str, object],
    duration_seconds: float,
) -> str:
    """Build a system-style instruction block for an LLM narrator.

    Word budget scales with on-screen segment duration. Style targets
    clinical dictation: crisp observations, numbers spelled out, and
    ellipses between distinct findings.
    """
    max_words = max_narration_words_for_duration(duration_seconds)
    summary_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(kinematic_summary.items())
    )
    return (
        "You are a surgical skills narrator producing a single spoken line for "
        "one gesture segment of a laparoscopic suturing trial.\n\n"
        "STYLE (mandatory):\n"
        "- Clinical dictation tone: concise, neutral, present tense, as if "
        "dictating into a microphone in the OR.\n"
        "- Spell out all numbers in words (e.g. \"three\" not \"3\").\n"
        "- Separate distinct observations with an ellipsis and space "
        "(\"... \") so the TTS inserts a short pause between clauses.\n"
        "- Do not include stage directions, quotes, or meta commentary.\n\n"
        f"GESTURE: {gesture_label}\n"
        f"DESCRIPTION: {gesture_description}\n"
        f"SEGMENT_DURATION_SECONDS: {float(duration_seconds):.3f}\n"
        f"MAX_WORDS (including spelled-out number words): {max_words}\n\n"
        "KINEMATIC_SUMMARY:\n"
        f"{summary_lines}\n\n"
        f"Respond with at most {max_words} words of narration text only, "
        "no preamble or bullet list."
    )


def _smoothness_label(motion_smoothness: float) -> str:
    if motion_smoothness >= 0.2:
        return "very smooth"
    if motion_smoothness >= 0.05:
        return "smooth"
    return "variable"


def render_narration_text(
    gesture_label: str,
    summary: Dict[str, float | str],
) -> str:
    desc = GESTURE_DESCRIPTIONS.get(gesture_label, f"Performing {gesture_label}")
    speed = str(summary.get("speed_rating", "optimal"))
    smooth = _smoothness_label(float(summary.get("motion_smoothness", 0.0)))
    g_left = float(summary.get("avg_gripper_left", 0.0))
    g_right = float(summary.get("avg_gripper_right", 0.0))
    lead = f"{desc}. Motion pace is {speed} and {smooth}."
    grip = (
        f"Left gripper angle averages {g_left:.1f}, "
        f"right gripper angle averages {g_right:.1f}."
    )
    return f"{lead} {grip}"


def build_narration_payload(
    gesture_label: str,
    kinematic_segment: np.ndarray,
    expert_speed_stats: Optional[Dict[str, SpeedStats | Dict[str, float]]] = None,
    min_count_for_empirical: int = 8,
) -> Dict[str, object]:
    summary = extract_kinematic_summary(
        kinematic_segment=kinematic_segment,
        gesture_label=gesture_label,
        expert_speed_stats=expert_speed_stats,
        min_count_for_empirical=min_count_for_empirical,
    )
    return {
        "gesture_label": gesture_label,
        "gesture_description": GESTURE_DESCRIPTIONS.get(
            gesture_label, f"Performing {gesture_label}"
        ),
        "summary": summary,
        "narration_text": render_narration_text(gesture_label, summary),
    }


def collapse_frame_records(
    per_frame_records: List[Dict[str, object]],
    int_to_gesture: Dict[int, str],
    fps_out: float,
) -> List[Dict[str, object]]:
    """Collapse per-frame records into contiguous gesture segments."""
    if not per_frame_records:
        return []
    out: List[Dict[str, object]] = []
    first = per_frame_records[0]
    cur: Dict[str, object] = {
        "start_output_index": int(first["output_index"]),
        "end_output_index": int(first["output_index"]),
        "gesture": (
            first.get("gesture")
            or int_to_gesture.get(int(first["gesture_int"]))
            or "G1"
        ),
        "gesture_int": int(first["gesture_int"]),
        "source_frames": [int(first["source_frame"])],
    }
    for rec in per_frame_records[1:]:
        gesture = (
            rec.get("gesture")
            or int_to_gesture.get(int(rec["gesture_int"]))
            or "G1"
        )
        gesture_int = int(rec["gesture_int"])
        output_index = int(rec["output_index"])
        if gesture == cur["gesture"] and gesture_int == cur["gesture_int"]:
            cur["end_output_index"] = output_index
            cur["source_frames"] = list(cur["source_frames"]) + [int(rec["source_frame"])]
            continue
        cur["start_time"] = float(cur["start_output_index"]) / fps_out
        cur["end_time"] = (float(cur["end_output_index"]) + 1.0) / fps_out
        out.append(cur)
        cur = {
            "start_output_index": output_index,
            "end_output_index": output_index,
            "gesture": gesture,
            "gesture_int": gesture_int,
            "source_frames": [int(rec["source_frame"])],
        }
    cur["start_time"] = float(cur["start_output_index"]) / fps_out
    cur["end_time"] = (float(cur["end_output_index"]) + 1.0) / fps_out
    out.append(cur)
    return out


def build_expert_speed_stats(
    data_root: str | Path,
    task: str = "suturing",
    min_segment_frames: int = 2,
) -> Dict[str, Dict[str, float]]:
    """Compute per-gesture speed stats from expert trials."""
    data_root = Path(data_root)
    task_dir = _resolve_ci(data_root / task)
    task_prefix = "Suturing" if task.lower() == "suturing" else task.capitalize()
    metafile = _resolve_ci(task_dir / f"meta_file_{task_prefix}.txt")
    if not metafile.exists():
        return {}
    expert_trials = filter_expert_trials(metafile)
    if not expert_trials:
        return {}

    gesture_speeds: Dict[str, List[float]] = {}
    for trial_name in expert_trials:
        suffix = trial_name.replace(f"{task_prefix}_", "")
        kin_path = _resolve_ci(task_dir / "kinematics" / "allgestures" / f"{task_prefix}_{suffix}.txt")
        tx_path = _resolve_ci(task_dir / "transcriptions" / f"{task_prefix}_{suffix}.txt")
        if not kin_path.exists() or not tx_path.exists():
            continue
        kin = parse_kinematics(kin_path)
        segments = parse_transcription(tx_path)
        for start, end, gesture in segments:
            s = max(0, int(start))
            e = min(int(end), kin.shape[0] - 1)
            if e < s:
                continue
            segment = kin[s : e + 1]
            if segment.shape[0] < min_segment_frames:
                continue
            avg_speed = float(np.nanmean(_speed_per_row(segment)))
            gesture_speeds.setdefault(gesture, []).append(avg_speed)

    stats: Dict[str, Dict[str, float]] = {}
    for gesture, values in gesture_speeds.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            continue
        stats[gesture] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "count": int(arr.size),
        }
    return stats
