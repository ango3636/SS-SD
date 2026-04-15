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
        min_box_area: int = 0,
    ) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Install `ultralytics` first.")
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.target_classes = target_classes
        self.min_box_area = min_box_area

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
