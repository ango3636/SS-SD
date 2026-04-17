from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class YoloDetector:
    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 640,
        target_classes: list[int] | None = None,
        target_class_names: list[str] | None = None,
        strict_target_class_names: bool = False,
        min_box_area: int = 0,
    ) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Install `ultralytics` first.")
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.target_classes = target_classes
        self.target_class_names = target_class_names or []
        self.strict_target_class_names = strict_target_class_names
        self.min_box_area = min_box_area
        self.model_names = self._normalize_names(getattr(self.model, "names", {}))
        self.target_classes = self._resolve_target_classes(self.target_classes, self.target_class_names)

    @staticmethod
    def _normalize_names(raw_names: Any) -> dict[int, str]:
        if isinstance(raw_names, dict):
            return {int(k): str(v) for k, v in raw_names.items()}
        if isinstance(raw_names, list):
            return {i: str(name) for i, name in enumerate(raw_names)}
        return {}

    def _resolve_target_classes(
        self,
        target_classes: list[int] | None,
        target_class_names: list[str],
    ) -> list[int] | None:
        if target_classes is not None:
            return target_classes
        if not target_class_names:
            return None

        normalized_targets = {name.strip().lower() for name in target_class_names if name.strip()}
        if not normalized_targets:
            return None

        matched_class_ids = [
            class_id
            for class_id, class_name in self.model_names.items()
            if class_name.strip().lower() in normalized_targets
        ]

        if not matched_class_ids and self.strict_target_class_names:
            available_names = ", ".join(sorted(self.model_names.values()))
            requested = ", ".join(sorted(normalized_targets))
            raise ValueError(
                "None of the requested target_class_names were found in the loaded model labels. "
                f"Requested: [{requested}]. Available model labels: [{available_names}]"
            )
        return matched_class_ids or None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=self.target_classes,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        names: dict[int, str] = getattr(results[0], "names", {}) or {}
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            det = Detection(
                class_id=cls_id,
                class_name=names.get(cls_id, str(cls_id)),
                confidence=conf,
                x1=xyxy[0],
                y1=xyxy[1],
                x2=xyxy[2],
                y2=xyxy[3],
            )
            if det.area >= self.min_box_area:
                detections.append(det)
        return detections
