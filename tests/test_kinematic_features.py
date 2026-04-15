import pandas as pd

from suturing_pipeline.kinematics.features import compute_kinematic_features


def test_compute_kinematic_features_columns():
    df = pd.DataFrame(
        {
            13: [0.2, 0.3, 0.4, 0.5, 0.6],
            14: [0.1, 0.2, 0.1, 0.2, 0.1],
            15: [0.0, 0.1, 0.1, 0.2, 0.2],
        }
    )
    out = compute_kinematic_features(df, translational_velocity_cols=[13, 14, 15], sampling_hz=30.0)
    assert len(out) == len(df)
    assert {"velocity", "acceleration", "jerk", "abs_jerk"}.issubset(out.columns)
