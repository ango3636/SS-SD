# Suturing Stable Diffusion Pipeline

Clean, modular pipeline for surgical suturing analysis and synthesis:

1. Data ingestion and modality alignment (video + kinematics + transcription)
2. Detection and overlay export
3. Kinematic feature engineering (velocity, acceleration, jerk)
4. ControlNet/SVD-ready synthesis scaffolding
5. Synchronized comparison dashboard

## Repository Layout

- `versions/` immutable notebook snapshots
- `notebooks/` phase-based exploratory notebooks
- `src/suturing_pipeline/` reusable Python package
- `scripts/` CLI entry points for each phase
- `configs/` project configuration files
- `tests/` initial unit tests

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run individual phases:

```bash
python scripts/prepare_trials.py --config configs/base.yaml
python scripts/run_detection.py --config configs/base.yaml
python scripts/compute_kinematics.py --config configs/base.yaml
python scripts/run_synthesis.py --config configs/base.yaml
python scripts/launch_dashboard.py --config configs/base.yaml
```

## Data Expectations

This project expects each trial to provide:

- `capture1` and/or `capture2` video
- kinematics file sampled at fixed rate (30 Hz for JIGSAWS)
- transcription/gesture timing file

`prepare_trials.py` discovers and links these modalities into a unified trial index.

## Current Scope

The baseline extraction from the original notebook includes:

- video metadata/frame extraction utilities
- YOLO detector wrapper + classical motion fallback detector
- detection export helpers (frames/crops/metadata)
- sequence dataset/model components for temporal prediction

The ControlNet/SVD integration is intentionally scaffolded and ready to connect to your trained models.
