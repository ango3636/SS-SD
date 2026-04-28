from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_kinematics_file(path: str | Path) -> pd.DataFrame:
    """Read a kinematics table from CSV/TSV/TXT."""
    kin_path = Path(path)
    if not kin_path.exists():
        raise FileNotFoundError(f"Kinematics file not found: {kin_path}")

    suffix = kin_path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(kin_path, sep=None, engine="python")
    else:
        try:
            df = pd.read_csv(kin_path)
        except pd.errors.ParserError:
            df = pd.read_csv(kin_path, sep=None, engine="python")

    if df.empty:
        raise ValueError(f"Kinematics file is empty: {kin_path}")
    return df


def align_kinematics_to_frames(
    kinematics_df: pd.DataFrame,
    frame_count: int,
    video_fps: float,
    kinematics_hz: float,
) -> pd.DataFrame:
    """
    Align kinematics samples to each video frame by nearest timestamp.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if video_fps <= 0 or kinematics_hz <= 0:
        raise ValueError("video_fps and kinematics_hz must be positive.")
    if kinematics_df.empty:
        raise ValueError("kinematics_df must contain at least one row.")

    frame_index = np.arange(frame_count, dtype=int)
    frame_time = frame_index / float(video_fps)
    kin_count = len(kinematics_df)
    kin_time = np.arange(kin_count, dtype=float) / float(kinematics_hz)

    raw_indices = np.rint(frame_time * float(kinematics_hz)).astype(int)
    kinematics_index = np.clip(raw_indices, 0, kin_count - 1)

    aligned = kinematics_df.iloc[kinematics_index].reset_index(drop=True).copy()
    aligned.insert(0, "kinematics_index", kinematics_index)
    aligned.insert(0, "frame_index", frame_index)
    aligned["frame_time_sec"] = frame_time
    aligned["kinematics_time_sec"] = kin_time[kinematics_index]
    return aligned
