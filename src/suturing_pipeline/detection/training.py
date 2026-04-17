from __future__ import annotations

from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


def _require_ultralytics() -> None:
    if YOLO is None:
        raise ImportError("ultralytics is not installed. Install `ultralytics` first.")


def write_yolo_dataset_yaml(
    dataset_root: str | Path,
    class_names: list[str],
    output_yaml_path: str | Path,
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_yaml_path = Path(output_yaml_path)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    test_images = dataset_root / "images" / "test"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"
    test_labels = dataset_root / "labels" / "test"

    if not train_images.exists():
        raise ValueError(f"Missing training image directory: {train_images}")
    if not val_images.exists():
        raise ValueError(f"Missing validation image directory: {val_images}")
    if not train_labels.exists():
        raise ValueError(f"Missing training label directory: {train_labels}")
    if not val_labels.exists():
        raise ValueError(f"Missing validation label directory: {val_labels}")
    if not class_names:
        raise ValueError("At least one class name is required.")

    payload: dict[str, object] = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    if test_images.exists() and test_labels.exists():
        payload["test"] = "images/test"

    with output_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return output_yaml_path


def train_yolo_detector(
    dataset_yaml_path: str | Path,
    base_weights: str,
    *,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = None,
    project_dir: str | Path = "./outputs/yolo_training",
    run_name: str = "needle_head_detector",
) -> Path:
    _require_ultralytics()
    dataset_yaml_path = Path(dataset_yaml_path).resolve()
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_weights)
    train_kwargs = {
        "data": str(dataset_yaml_path),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": str(project_dir),
        "name": run_name,
    }
    if device is not None and str(device).strip():
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        candidate = project_dir / run_name / "weights" / "best.pt"
        return candidate
    return Path(save_dir) / "weights" / "best.pt"
