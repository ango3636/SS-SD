import numpy as np

from suturing_pipeline.audio.narration_templates import (
    GESTURE_DESCRIPTIONS,
    build_narration_payload,
    extract_kinematic_summary,
)


def _segment_with_speed(speed: float, n: int = 12) -> np.ndarray:
    seg = np.zeros((n, 76), dtype=np.float32)
    # Master left / master right translational velocity (JIGSAWS 1-based 13–15 / 32–34)
    seg[:, 12] = speed
    seg[:, 31] = speed
    seg[:, 56] = 10.0
    seg[:, 75] = 12.0
    return seg


def test_gesture_descriptions_has_expected_keys():
    for i in range(1, 16):
        key = f"G{i}"
        assert key in GESTURE_DESCRIPTIONS
        assert isinstance(GESTURE_DESCRIPTIONS[key], str)
        assert GESTURE_DESCRIPTIONS[key]


def test_extract_kinematic_summary_uses_absolute_fallback():
    seg = _segment_with_speed(20.0)
    summary = extract_kinematic_summary(seg, gesture_label="G1", expert_speed_stats={})
    assert summary["speed_rating"] == "slow"
    assert summary["speed_rating_source"] == "absolute_fallback"
    assert summary["segment_length_frames"] == 12


def test_extract_kinematic_summary_uses_empirical_stats():
    seg = _segment_with_speed(50.0)
    stats = {"G1": {"mean": 60.0, "std": 5.0, "count": 20}}
    summary = extract_kinematic_summary(
        seg,
        gesture_label="G1",
        expert_speed_stats=stats,
        min_count_for_empirical=8,
    )
    assert summary["speed_rating"] == "slow"
    assert summary["speed_rating_source"] == "empirical"


def test_build_narration_payload_contains_text_and_summary():
    seg = _segment_with_speed(70.0)
    payload = build_narration_payload("G3", seg)
    assert payload["gesture_label"] == "G3"
    assert "summary" in payload
    assert "narration_text" in payload
    assert "Pushing needle through tissue" in payload["narration_text"]
