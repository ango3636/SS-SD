from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def vector_magnitude(df: pd.DataFrame, cols: list[int] | list[str]) -> np.ndarray:
    if all(isinstance(c, int) for c in cols):
        # Support both integer-labeled columns and positional indices.
        if all(c in df.columns for c in cols):
            arr = df.loc[:, cols].to_numpy(dtype=float)
        else:
            arr = df.iloc[:, cols].to_numpy(dtype=float)
    else:
        arr = df.loc[:, cols].to_numpy(dtype=float)
    return np.linalg.norm(arr, axis=1)


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    return np.gradient(x, dt)


def smooth_signal(x: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    if len(x) < 5:
        return x
    window_length = min(window_length, len(x) if len(x) % 2 == 1 else len(x) - 1)
    window_length = max(5, window_length)
    if window_length % 2 == 0:
        window_length += 1
    polyorder = min(polyorder, window_length - 1)
    return savgol_filter(x, window_length=window_length, polyorder=polyorder)


def compute_kinematic_features(
    kinematics_df: pd.DataFrame,
    translational_velocity_cols: list[int] | list[str],
    sampling_hz: float = 30.0,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
) -> pd.DataFrame:
    dt = 1.0 / sampling_hz
    velocity = vector_magnitude(kinematics_df, translational_velocity_cols)
    velocity_smooth = smooth_signal(velocity, smooth_window, smooth_polyorder)
    acceleration = finite_difference(velocity_smooth, dt)
    jerk = finite_difference(acceleration, dt)

    return pd.DataFrame(
        {
            "step_index": np.arange(len(velocity), dtype=int),
            "velocity": velocity,
            "velocity_smooth": velocity_smooth,
            "acceleration": acceleration,
            "jerk": jerk,
            "abs_jerk": np.abs(jerk),
        }
    )
