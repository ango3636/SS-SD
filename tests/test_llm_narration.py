"""Tests for LLM narration helpers (no live HTTP)."""

from __future__ import annotations

import numpy as np

from suturing_pipeline.audio.llm_narration import (
    apply_llm_narration_to_segments,
    synthesize_narration_line,
)
from suturing_pipeline.audio.narration_templates import (
    extract_kinematic_summary,
    render_narration_text,
)


def test_synthesize_template_matches_render() -> None:
    kin = np.zeros((10, 76), dtype=np.float64)
    summary = extract_kinematic_summary(kin, "G1")
    via_llm = synthesize_narration_line(
        gesture_label="G1",
        gesture_description="test",
        kinematic_summary=summary,
        duration_seconds=3.0,
        backend="template",
    )
    direct = render_narration_text("G1", summary)
    assert via_llm == direct


def test_apply_llm_template_is_noop_copy_semantics() -> None:
    seg = {
        "start_time": 0.0,
        "end_time": 1.0,
        "gesture": "G1",
        "gesture_description": "d",
        "summary": extract_kinematic_summary(np.zeros((5, 76)), "G1"),
        "narration_text": "original",
    }
    out = apply_llm_narration_to_segments([seg], backend="template")
    assert len(out) == 1
    assert out[0]["narration_text"] == "original"
