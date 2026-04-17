from pathlib import Path

from suturing_pipeline.data.loader import discover_trials


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_discover_trials_local_multitask(tmp_path: Path):
    # suturing trial
    _touch(tmp_path / "suturing" / "video" / "Suturing_B001_capture1.avi")
    _touch(tmp_path / "suturing" / "video" / "Suturing_B001_capture2.avi")
    _touch(tmp_path / "suturing" / "kinematics" / "Suturing_B001_kinematics.csv")
    _touch(tmp_path / "suturing" / "transcription" / "Suturing_B001_transcription.txt")
    # knot tying trial
    _touch(tmp_path / "knot tying" / "video" / "Knot_B002_capture1.avi")
    _touch(tmp_path / "knot tying" / "kinematics" / "Knot_B002_kinematics.csv")

    df = discover_trials(
        data_root=tmp_path,
        ingestion_config={"source": "local", "tasks": []},
    )

    assert len(df) == 2
    assert set(df["task_name"]) == {"suturing", "knot tying"}
    assert set(df["data_source"]) == {"local"}
    sut_row = df[df["task_name"] == "suturing"].iloc[0]
    assert sut_row["video_capture1"].endswith("capture1.avi")
    assert sut_row["video_capture2"].endswith("capture2.avi")
    assert sut_row["kinematics_path"].endswith("kinematics.csv")
    assert sut_row["transcription_path"].endswith("transcription.txt")


def test_discover_trials_local_task_filter(tmp_path: Path):
    _touch(tmp_path / "suturing" / "video" / "Suturing_B001_capture1.avi")
    _touch(tmp_path / "knot tying" / "video" / "Knot_B002_capture1.avi")

    df = discover_trials(
        data_root=tmp_path,
        ingestion_config={"source": "local", "tasks": ["suturing"]},
    )

    assert len(df) == 1
    assert df.iloc[0]["task_name"] == "suturing"
