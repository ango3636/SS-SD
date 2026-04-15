from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.config import ensure_output_dirs, load_config
from suturing_pipeline.detection.export import save_detection_artifacts, save_detection_metadata
from suturing_pipeline.detection.motion_detector import ClassicalMotionDetector
from suturing_pipeline.detection.yolo_detector import YoloDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frame-level detection and export artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", default=None, help="Optional direct video path override")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    output_root = Path(cfg.paths["output_root"]) / "detection"

    trial_df = pd.read_csv(cfg.paths["trial_index_csv"]) if Path(cfg.paths["trial_index_csv"]).exists() else pd.DataFrame()
    video_path = args.video
    if video_path is None and not trial_df.empty:
        row0 = trial_df.iloc[0]
        video_path = row0.get("video_capture2") or row0.get("video_capture1")
    if not video_path:
        raise ValueError("No video path supplied and no trial index video available.")

    use_yolo = bool(cfg.detection.get("use_yolo", True))
    if use_yolo:
        detector = YoloDetector(
            weights=cfg.detection.get("yolo_weights", "yolov8n.pt"),
            conf=float(cfg.detection.get("conf_threshold", 0.25)),
            iou=float(cfg.detection.get("iou_threshold", 0.45)),
            img_size=int(cfg.detection.get("img_size", 640)),
            min_box_area=int(cfg.detection.get("min_box_area", 0)),
        )
    else:
        detector = ClassicalMotionDetector(
            min_area=int(cfg.raw.get("motion_fallback", {}).get("min_area", 1200)),
            blur_kernel=int(cfg.raw.get("motion_fallback", {}).get("blur_kernel", 7)),
            threshold_binary=int(cfg.raw.get("motion_fallback", {}).get("threshold_binary", 25)),
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    rows = []
    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        if detections:
            rows.extend(
                save_detection_artifacts(
                    frame_idx=frame_idx,
                    frame=frame,
                    detections=detections,
                    output_root=output_root,
                    save_crops=bool(cfg.detection.get("save_crops_if_detected", True)),
                    save_annotated=bool(cfg.detection.get("save_annotated_frames", True)),
                )
            )
        frame_idx += 1
    cap.release()

    save_detection_metadata(rows, cfg.paths["detection_metadata_csv"], output_root / "detections.json")
    print(f"Processed frames: {frame_idx}")
    print(f"Detection rows: {len(rows)}")


if __name__ == "__main__":
    main()
