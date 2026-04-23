from suturing_pipeline.audio.narration_templates import collapse_frame_records


def test_collapse_frame_records_groups_contiguous_gesture_runs():
    frames = [
        {"output_index": 0, "source_frame": 10, "gesture": "G1", "gesture_int": 1},
        {"output_index": 1, "source_frame": 16, "gesture": "G1", "gesture_int": 1},
        {"output_index": 2, "source_frame": 22, "gesture": "G2", "gesture_int": 2},
        {"output_index": 3, "source_frame": 28, "gesture": "G2", "gesture_int": 2},
        {"output_index": 4, "source_frame": 34, "gesture": "G1", "gesture_int": 1},
    ]
    out = collapse_frame_records(frames, int_to_gesture={1: "G1", 2: "G2"}, fps_out=5.0)
    assert len(out) == 3
    assert out[0]["gesture"] == "G1"
    assert out[0]["start_time"] == 0.0
    assert out[0]["end_time"] == 0.4
    assert out[1]["gesture"] == "G2"
    assert out[1]["source_frames"] == [22, 28]
    assert out[2]["gesture"] == "G1"


def test_collapse_frame_records_recovers_label_from_int_mapping():
    frames = [
        {"output_index": 0, "source_frame": 1, "gesture": None, "gesture_int": 3},
        {"output_index": 1, "source_frame": 2, "gesture": None, "gesture_int": 3},
    ]
    out = collapse_frame_records(frames, int_to_gesture={3: "G3"}, fps_out=10.0)
    assert len(out) == 1
    assert out[0]["gesture"] == "G3"
