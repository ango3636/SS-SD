from __future__ import annotations

import cv2


def put_top_left_text(frame, text: str, color=(255, 255, 255)):
    out = frame.copy()
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out
