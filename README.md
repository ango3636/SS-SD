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

Prepare a labeling set from your `trial_index.csv` videos (works with your
Google Drive materialized cache paths):

```bash
python scripts/prepare_labeling_set.py \
  --trial-index ./outputs/trial_index.csv \
  --output-root ./data/suturing_yolo \
  --tasks "Suturing,Needle_Passing" \
  --capture capture2 \
  --frames-per-trial 30 \
  --sampling uniform
```

This creates:

- `data/suturing_yolo/images/train/*.jpg`
- `data/suturing_yolo/images/val/*.jpg`
- `data/suturing_yolo/labels/train/*.txt` (empty placeholders)
- `data/suturing_yolo/labels/val/*.txt` (empty placeholders)
- `data/suturing_yolo/manifests/labeling_manifest.csv`

Labeling workflow:

1. Import `images/train` and `images/val` into CVAT or Label Studio.
2. Annotate with your class list (for example: `needle_head`).
3. Export YOLO labels and place `.txt` files back into matching `labels/train`
   and `labels/val` paths with the same stem as each image.
4. Train with `scripts/train_yolo.py`.

Train a custom YOLO detector (from labeled suturing frames in YOLO format):

```bash
python scripts/train_yolo.py \
  --dataset-root ./data/suturing_yolo \
  --classes "needle_head,needle_driver,forceps,thread,left_tool,right_tool,tissue_region" \
  --base-weights yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

Expected dataset structure for training:

- `data/suturing_yolo/images/train/*.jpg`
- `data/suturing_yolo/images/val/*.jpg`
- `data/suturing_yolo/labels/train/*.txt`
- `data/suturing_yolo/labels/val/*.txt`

After training, set `detection.yolo_weights` in `configs/base.yaml` to the printed
`best.pt` path and keep `detection.target_class_names` aligned to your trained labels.

### Detection Model Note

The default `yolov8n.pt` is a generic COCO model and will not detect
suturing-specific targets (for example, needle heads) reliably. Point
`detection.yolo_weights` to a domain-trained model whose labels include your
needle class names, and configure:

- `detection.target_class_names` (for example: `["needle_head"]`)
- `detection.strict_target_class_names: true` to fail fast when labels do not match

## Google Drive OAuth Setup (Primary Ingestion Path)

The default config now uses Google Drive API access for trial discovery.

1. Place your downloaded OAuth client secret JSON at:
   - `./secrets/google_oauth_client.json`
2. Keep `configs/base.yaml` as-is or set:
   - `ingestion.source: gdrive`
   - `ingestion.gdrive.credentials_path` to your JSON file
   - `ingestion.gdrive.oauth_flow: console` (prints auth URL in terminal; no auto-open browser)
   - optional `ingestion.gdrive.root_folder_id` (recommended if folder names are ambiguous)
   - optional `ingestion.gdrive.materialize_local: true` to download indexed files into cache
   - optional `ingestion.gdrive.cache_dir` for local cache location
3. Run ingestion:

```bash
python scripts/prepare_trials.py --config configs/base.yaml
```

On first run with `oauth_flow: console`, the script prints a Google consent URL in
terminal; open it manually in your browser and complete consent. The refresh token
is saved at `./secrets/google_token.json` and reused on later runs.

By default, discovered Drive files are cached locally under `./data/gdrive_cache`
so downstream scripts can read normal file paths from `trial_index.csv`.

Fallback mode is still supported by setting `ingestion.source: local` and pointing
`paths.data_root` to a local dataset mirror.

## Data Expectations

This project expects each trial to provide:

- `capture1` and/or `capture2` video
- kinematics file sampled at fixed rate (30 Hz for JIGSAWS)
- transcription/gesture timing file

`prepare_trials.py` discovers and links these modalities into a unified trial index.
For JIGSAWS-style layouts, discovery expects task folders containing `video/`,
`kinematics/`, and `transcription/` subfolders.

## Current Scope

The baseline extraction from the original notebook includes:

- video metadata/frame extraction utilities
- YOLO detector wrapper + classical motion fallback detector
- detection export helpers (frames/crops/metadata)
- sequence dataset/model components for temporal prediction

The ControlNet/SVD integration is intentionally scaffolded and ready to connect to your trained models.
