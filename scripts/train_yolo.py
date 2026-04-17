from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.detection.training import train_yolo_detector, write_yolo_dataset_yaml


def _parse_class_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",")]
    names = [name for name in names if name]
    if not names:
        raise ValueError("No class names provided.")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO detector for suturing instruments.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing images/train, images/val, labels/train, labels/val (YOLO format).",
    )
    parser.add_argument(
        "--classes",
        required=True,
        help='Comma-separated class names in index order, e.g. "needle_head,needle_driver,forceps".',
    )
    parser.add_argument(
        "--dataset-yaml",
        default="./outputs/yolo_training/suturing_dataset.yaml",
        help="Where to write the generated YOLO dataset YAML.",
    )
    parser.add_argument("--base-weights", default="yolov8n.pt", help="Base YOLO checkpoint to finetune.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help='Training device, e.g. "0", "cpu", "mps".')
    parser.add_argument("--project-dir", default="./outputs/yolo_training")
    parser.add_argument("--run-name", default="needle_head_detector")
    args = parser.parse_args()

    class_names = _parse_class_names(args.classes)
    dataset_yaml_path = write_yolo_dataset_yaml(
        dataset_root=args.dataset_root,
        class_names=class_names,
        output_yaml_path=args.dataset_yaml,
    )
    print(f"Saved dataset YAML: {dataset_yaml_path}")
    print(f"Classes: {class_names}")

    best_weights_path = train_yolo_detector(
        dataset_yaml_path=dataset_yaml_path,
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project_dir=args.project_dir,
        run_name=args.run_name,
    )
    print(f"Training complete. Best weights: {best_weights_path}")
    print("Next: set detection.yolo_weights to this best.pt in configs/base.yaml")


if __name__ == "__main__":
    main()
