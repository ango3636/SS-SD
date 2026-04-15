from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.config import ensure_output_dirs, load_config
from suturing_pipeline.synthesis.controlnet_pipeline import ControlNetSynthesisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ControlNet/SVD synthesis scaffold.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    runner = ControlNetSynthesisPipeline()
    summary = runner.run(
        detection_metadata_csv=cfg.paths["detection_metadata_csv"],
        kinematics_features_csv=cfg.paths["kinematics_features_csv"],
        output_dir=cfg.paths["synthesis_output_dir"],
    )
    print(summary)


if __name__ == "__main__":
    main()
