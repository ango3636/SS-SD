from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.detection.labeling import prepare_labeling_dataset


def _parse_tasks(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a YOLO-style labeling workspace from trial_index.csv videos."
    )
    parser.add_argument(
        "--trial-index",
        default="./outputs/trial_index.csv",
        help="Path to trial_index.csv (local paths or materialized gdrive cache paths).",
    )
    parser.add_argument(
        "--output-root",
        default="./data/suturing_yolo",
        help="Output dataset root (images/train|val + labels/train|val + manifests).",
    )
    parser.add_argument(
        "--tasks",
        default="Suturing",
        help='Comma-separated task filter, e.g. "Suturing,Needle_Passing". Empty means all tasks.',
    )
    parser.add_argument(
        "--capture",
        default="capture2",
        choices=["capture1", "capture2"],
        help="Preferred camera stream; script falls back to the other capture if missing.",
    )
    parser.add_argument("--frames-per-trial", type=int, default=30)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-frame", type=int, default=0)
    parser.add_argument("--max-frame", type=int, default=None)
    parser.add_argument("--sampling", default="uniform", choices=["uniform", "random"])
    args = parser.parse_args()

    manifest = prepare_labeling_dataset(
        trial_index_csv=args.trial_index,
        output_root=args.output_root,
        task_names=_parse_tasks(args.tasks),
        capture_preference=args.capture,
        frames_per_trial=args.frames_per_trial,
        max_trials=args.max_trials,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_frame=args.min_frame,
        max_frame=args.max_frame,
        sampling=args.sampling,
    )

    print(f"Saved labeling manifest rows: {len(manifest)}")
    print(f"Manifest path: {Path(args.output_root) / 'manifests' / 'labeling_manifest.csv'}")
    if not manifest.empty:
        print("Split counts:")
        print(manifest["split"].value_counts().to_string())
    print("Next: import images into CVAT/Label Studio and save YOLO txt labels in matching labels/<split>/ files.")


if __name__ == "__main__":
    main()
