import numpy as np
import pandas as pd

from suturing_pipeline.kinematics.features import compute_kinematic_features
from suturing_pipeline.kinematics.motion_columns import (
    MOTION_FEATURE_DIM,
    append_motion_columns,
)


def test_compute_kinematic_features_columns():
    df = pd.DataFrame(
        {
            12: [0.2, 0.3, 0.4, 0.5, 0.6],
            13: [0.1, 0.2, 0.1, 0.2, 0.1],
            14: [0.0, 0.1, 0.1, 0.2, 0.2],
        }
    )
    out = compute_kinematic_features(
        df, translational_velocity_cols=[12, 13, 14], sampling_hz=30.0
    )
    assert len(out) == len(df)
    assert {"velocity", "acceleration", "jerk", "abs_jerk"}.issubset(out.columns)


def test_append_motion_columns_shape():
    kin = np.zeros((10, 76), dtype=np.float64)
    aug = append_motion_columns(kin)
    assert aug.shape == (10, 76 + MOTION_FEATURE_DIM)
