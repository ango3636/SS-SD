from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.config import ensure_output_dirs, load_config
from suturing_pipeline.data.loader import discover_trials, save_trial_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and index multimodal surgical trials.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    ingestion = cfg.ingestion
    df = discover_trials(
        data_root=cfg.paths["data_root"],
        video_extensions=tuple(ingestion.get("video_extensions", [".avi", ".mp4", ".mov"])),
        kinematics_keywords=tuple(ingestion.get("kinematics_keywords", ["kinematics"])),
        transcription_keywords=tuple(ingestion.get("transcription_keywords", ["transcription"])),
        ingestion_config=ingestion,
    )
    save_trial_index(df, cfg.paths["trial_index_csv"])
    print(f"Discovered {len(df)} trials.")
    print(f"Saved index: {cfg.paths['trial_index_csv']}")


if __name__ == "__main__":
    main()
