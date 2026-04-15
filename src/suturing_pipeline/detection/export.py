from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd

from .yolo_detector import Detection


def draw_detections(frame, detections: Iterable[Detection]):
    out = frame.copy()
    for det in detections:
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        label = f"{det.class_name}:{det.confidence:.2f}"
        cv2.putText(out, label, (det.x1, max(20, det.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def save_detection_artifacts(
    frame_idx: int,
    frame,
    detections: list[Detection],
    output_root: str | Path,
    save_crops: bool = True,
    save_annotated: bool = True,
) -> list[dict]:
    output_root = Path(output_root)
    frames_dir = output_root / "frames"
    crops_dir = output_root / "crops"
    annotated_dir = output_root / "annotated"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    frame_name = f"frame_{frame_idx:06d}.jpg"
    frame_path = frames_dir / frame_name
    cv2.imwrite(str(frame_path), frame)

    if save_annotated:
        annotated = draw_detections(frame, detections)
        cv2.imwrite(str(annotated_dir / frame_name), annotated)

    rows: list[dict] = []
    for i, det in enumerate(detections):
        crop_path = None
        if save_crops:
            crop = frame[det.y1:det.y2, det.x1:det.x2]
            crop_name = f"frame_{frame_idx:06d}_det_{i:02d}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

        rows.append(
            {
                "frame_index": frame_idx,
                "frame_path": str(frame_path),
                "det_index": i,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "x1": det.x1,
                "y1": det.y1,
                "x2": det.x2,
                "y2": det.y2,
                "box_area": det.area,
                "crop_path": str(crop_path) if crop_path is not None else "",
            }
        )
    return rows


def save_detection_metadata(rows: list[dict], out_csv: str | Path, out_json: str | Path | None = None) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
