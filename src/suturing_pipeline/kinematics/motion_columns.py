"""Per-frame scalar motion features appended to raw JIGSAWS kinematics rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from suturing_pipeline.data.jigsaws_kinematics_layout import (
    TRANSLATIONAL_VELOCITY_COL_INDICES,
)

from .features import compute_kinematic_features

# Concatenated after the 76 raw columns when ``append_motion_features`` is enabled.
MOTION_FEATURE_DIM = 4


def per_frame_motion_features(kin_arr: np.ndarray) -> np.ndarray:
    """Return ``(T, 4)`` velocity / smoothed velocity / acceleration / jerk.

    Parameters
    ----------
    kin_arr:
        Raw kinematics array ``(T, 76)`` as loaded by ``parse_kinematics``.
    """
    df = pd.DataFrame(kin_arr)
    feat = compute_kinematic_features(
        kinematics_df=df,
        translational_velocity_cols=TRANSLATIONAL_VELOCITY_COL_INDICES,
    )
    out = feat[["velocity", "velocity_smooth", "acceleration", "jerk"]].to_numpy(
        dtype=np.float64
    )
    if out.shape[0] != kin_arr.shape[0]:
        raise ValueError(
            f"motion feature length {out.shape[0]} != kin length {kin_arr.shape[0]}"
        )
    return out


def append_motion_columns(kin_arr: np.ndarray) -> np.ndarray:
    """Return ``(T, 76 + MOTION_FEATURE_DIM)`` by concatenating derivatives."""
    extra = per_frame_motion_features(kin_arr)
    return np.concatenate([kin_arr, extra], axis=1)
