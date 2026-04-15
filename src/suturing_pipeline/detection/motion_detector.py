from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .yolo_detector import Detection


class ClassicalMotionDetector:
    def __init__(
        self,
        min_area: int = 1200,
        blur_kernel: int = 7,
        threshold_binary: int = 25,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.threshold_binary = threshold_binary
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=16)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        fg = self.bg_subtractor.apply(frame)
        blur = cv2.GaussianBlur(fg, (self.blur_kernel, self.blur_kernel), 0)
        _, thresh = cv2.threshold(blur, self.threshold_binary, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area:
                continue
            detections.append(
                Detection(
                    class_id=0,
                    class_name="motion_roi",
                    confidence=1.0,
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                )
            )
        return detections
