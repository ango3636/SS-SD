"""Narration segment ordering and pad-to-min (no voice truncation)."""

import numpy as np

from suturing_pipeline.audio.tts_converter import (
    _ordered_valid_narration_segments,
    _pad_waveform_to_min_samples,
)


def _seg(start: float, end: float, text: str = "x", gesture: str = "G1") -> dict:
    return {
        "start_time": start,
        "end_time": end,
        "narration_text": text,
        "gesture": gesture,
    }


def test_ordered_preserves_non_overlapping_sort() -> None:
    a = _seg(1.0, 2.0, "one")
    b = _seg(0.0, 0.5, "two")
    out = _ordered_valid_narration_segments([a, b], min_segment_seconds=0.35)
    assert [s["narration_text"] for s in out] == ["two", "one"]


def test_ordered_delays_not_applied_here_only_sort() -> None:
    """Ordering is by annotation time; overlap avoidance is in build_trial mix_cursor."""
    late = _seg(5.0, 6.0, "late")
    early = _seg(0.0, 1.0, "early")
    out = _ordered_valid_narration_segments([late, early], min_segment_seconds=0.35)
    assert [s["narration_text"] for s in out] == ["early", "late"]


def test_ordered_drops_empty_text() -> None:
    out = _ordered_valid_narration_segments(
        [_seg(0.0, 1.0, " "), _seg(1.0, 2.0, "ok")],
        min_segment_seconds=0.35,
    )
    assert len(out) == 1
    assert out[0]["narration_text"] == "ok"


def test_pad_extends_short_waveform() -> None:
    w = np.array([0.5, -0.5], dtype=np.float32)
    p = _pad_waveform_to_min_samples(w, 6)
    assert p.shape == (6,)
    assert np.allclose(p[:2], w)
    assert float(np.max(np.abs(p[2:]))) == 0.0


def test_pad_does_not_truncate_long_waveform() -> None:
    w = np.ones(10, dtype=np.float32)
    p = _pad_waveform_to_min_samples(w, 4)
    assert p.shape == (10,)
    assert np.allclose(p, w)
