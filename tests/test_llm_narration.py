"""Tests for LLM narration helpers (no live HTTP)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from suturing_pipeline.audio.llm_narration import (
    apply_llm_narration_to_segments,
    synthesize_narration_line,
)
from suturing_pipeline.audio.narration_templates import (
    extract_kinematic_summary,
    kinematics_segment_to_jsonable,
    render_narration_text,
    write_narration_transcript,
)


def test_synthesize_template_matches_render() -> None:
    kin = np.zeros((10, 76), dtype=np.float64)
    summary = extract_kinematic_summary(kin, "G1")
    via_llm, prompt = synthesize_narration_line(
        gesture_label="G1",
        gesture_description="test",
        kinematic_summary=summary,
        duration_seconds=3.0,
        backend="template",
    )
    direct = render_narration_text("G1", summary)
    assert via_llm == direct
    assert prompt is None


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
    assert "llm_user_prompt" not in out[0]


@patch(
    "suturing_pipeline.audio.llm_narration.complete_narration_ollama",
    return_value="Spoken line from mock.",
)
def test_apply_llm_ollama_adds_prompt_and_template(_mock_ollama: object) -> None:
    seg = {
        "start_time": 0.0,
        "end_time": 2.0,
        "gesture": "G1",
        "gesture_description": "d",
        "summary": extract_kinematic_summary(np.zeros((5, 76)), "G1"),
        "narration_text": "template line here",
    }
    out = apply_llm_narration_to_segments(
        [seg],
        backend="ollama",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3.2",
    )
    assert len(out) == 1
    assert out[0]["narration_text_template"] == "template line here"
    assert out[0]["narration_text"] == "Spoken line from mock."
    assert out[0]["llm_user_prompt"]
    assert "KINEMATIC_SUMMARY" in out[0]["llm_user_prompt"]


def test_kinematics_segment_to_jsonable_roundtrip_shape() -> None:
    kin = np.array([[1.23456789, 0.0], [2.0, 3.999999]], dtype=np.float64)
    j = kinematics_segment_to_jsonable(kin)
    assert j == [[1.234568, 0.0], [2.0, 3.999999]]


def test_write_narration_transcript(tmp_path) -> None:
    path = tmp_path / "t.txt"
    write_narration_transcript(
        [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "gesture": "G1",
                "narration_text": "hello",
            }
        ],
        path,
    )
    assert "[0.000s – 1.000s] G1: hello" in path.read_text(encoding="utf-8")
