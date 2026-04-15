from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def run_dashboard(
    detection_csv: str | Path,
    kinematics_csv: str | Path,
    synthesis_dir: str | Path,
    app_title: str = "Suturing Comparison Dashboard",
) -> None:
    st.set_page_config(page_title=app_title, layout="wide")
    st.title(app_title)

    d_path = Path(detection_csv)
    k_path = Path(kinematics_csv)
    s_dir = Path(synthesis_dir)

    detections = pd.read_csv(d_path) if d_path.exists() else pd.DataFrame()
    kinematics = pd.read_csv(k_path) if k_path.exists() else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Raw Video")
        st.info("Bind raw video player here.")
    with c2:
        st.subheader("Detection Overlay")
        st.metric("Detected frames", int(detections["frame_index"].nunique()) if not detections.empty else 0)
    with c3:
        st.subheader("Synthesis (Optimal)")
        st.write(f"Output folder: `{s_dir}`")

    st.markdown("---")
    st.subheader("Kinematic Signals")
    if kinematics.empty:
        st.warning("No kinematics features found yet.")
    else:
        plot_cols = [c for c in ["velocity_smooth", "acceleration", "jerk"] if c in kinematics.columns]
        st.line_chart(kinematics[plot_cols])

    st.subheader("Gesture Timeline")
    st.info("Hook transcription-aligned gesture sequence here.")

    st.subheader("GRS Radar")
    st.info("Hook GRS score visualization here.")


if __name__ == "__main__":
    detection_csv = st.sidebar.text_input("Detection CSV", "./outputs/detection/detections.csv")
    kinematics_csv = st.sidebar.text_input("Kinematics CSV", "./outputs/kinematics/features.csv")
    synthesis_dir = st.sidebar.text_input("Synthesis Output Dir", "./outputs/synthesis")
    run_dashboard(detection_csv=detection_csv, kinematics_csv=kinematics_csv, synthesis_dir=synthesis_dir)
