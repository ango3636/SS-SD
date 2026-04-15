from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.config import ensure_output_dirs, load_config
from suturing_pipeline.data.alignment import read_kinematics_file
from suturing_pipeline.kinematics.features import compute_kinematic_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute velocity/acceleration/jerk features from kinematics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--kinematics", default=None, help="Optional direct kinematics file override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    kin_path = args.kinematics
    if kin_path is None and Path(cfg.paths["trial_index_csv"]).exists():
        trial_df = pd.read_csv(cfg.paths["trial_index_csv"])
        if not trial_df.empty:
            kin_path = trial_df.iloc[0].get("kinematics_path")
    if not kin_path:
        raise ValueError("No kinematics file provided/found.")

    kin_df = read_kinematics_file(kin_path)
    col_spec = cfg.kinematics.get("translational_velocity_cols", [13, 14, 15])
    cols = [int(c) for c in col_spec]

    features_df = compute_kinematic_features(
        kinematics_df=kin_df,
        translational_velocity_cols=cols,
        sampling_hz=float(cfg.kinematics.get("sampling_hz", 30.0)),
        smooth_window=int(cfg.kinematics.get("savgol_window", 11)),
        smooth_polyorder=int(cfg.kinematics.get("savgol_polyorder", 3)),
    )
    out_csv = Path(cfg.paths["kinematics_features_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(features_df.head())


if __name__ == "__main__":
    main()
