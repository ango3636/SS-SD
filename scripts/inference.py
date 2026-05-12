"""Compatibility shim: single-frame SD inference.

The canonical implementation is ``inference_sd.py``. For narration,
AudioGen OR ambience, and video export, use ``generate_eval_video.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "inference_sd.py"),
        run_name="__main__",
    )
