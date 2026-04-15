import pandas as pd

from suturing_pipeline.data.alignment import align_kinematics_to_frames


def test_align_kinematics_to_frames_shape():
    kin = pd.DataFrame({"vx": [0.1, 0.2, 0.3, 0.4], "vy": [0.0, 0.0, 0.0, 0.0]})
    aligned = align_kinematics_to_frames(kinematics_df=kin, frame_count=3, video_fps=30.0, kinematics_hz=30.0)
    assert len(aligned) == 3
    assert "frame_index" in aligned.columns
    assert "kinematics_index" in aligned.columns
