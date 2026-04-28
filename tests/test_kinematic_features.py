import pandas as pd

from suturing_pipeline.kinematics.features import compute_kinematic_features


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
