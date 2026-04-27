"""Narration segment scheduling (non-overlapping mix bus)."""

from suturing_pipeline.audio.tts_converter import _schedule_narration_segments


def _seg(start: float, end: float, text: str = "x", gesture: str = "G1") -> dict:
    return {
        "start_time": start,
        "end_time": end,
        "narration_text": text,
        "gesture": gesture,
    }


def test_schedule_preserves_non_overlapping_order() -> None:
    a = _seg(0.0, 1.0, "one")
    b = _seg(1.0, 2.0, "two")
    out = _schedule_narration_segments([a, b], min_segment_seconds=0.35)
    assert len(out) == 2
    assert out[0][:2] == (0.0, 1.0)
    assert out[1][:2] == (1.0, 2.0)


def test_schedule_delays_second_when_intervals_overlap() -> None:
    a = _seg(0.0, 10.0, "first")
    b = _seg(5.0, 15.0, "second")
    out = _schedule_narration_segments([b, a], min_segment_seconds=0.35)
    assert len(out) == 2
    assert out[0][0] == 0.0 and out[0][1] == 10.0
    assert out[1][0] == 10.0
    assert out[1][1] == 15.0


def test_schedule_sorts_by_start_time() -> None:
    late = _seg(5.0, 6.0, "late")
    early = _seg(0.0, 1.0, "early")
    out = _schedule_narration_segments([late, early], min_segment_seconds=0.35)
    assert [o[2]["narration_text"] for o in out] == ["early", "late"]


def test_schedule_drops_empty_text() -> None:
    out = _schedule_narration_segments(
        [_seg(0.0, 1.0, " "), _seg(1.0, 2.0, "ok")],
        min_segment_seconds=0.35,
    )
    assert len(out) == 1
    assert out[0][2]["narration_text"] == "ok"
