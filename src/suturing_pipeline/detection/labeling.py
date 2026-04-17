from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip().lower())


def _normalize_optional_path(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def choose_video_path(row: pd.Series, capture_preference: str = "capture2") -> tuple[str, str]:
    capture_preference = capture_preference.lower()
    video_capture1 = _normalize_optional_path(row.get("video_capture1", ""))
    video_capture2 = _normalize_optional_path(row.get("video_capture2", ""))

    if capture_preference == "capture1":
        if video_capture1:
            return video_capture1, "capture1"
        if video_capture2:
            return video_capture2, "capture2"
    else:
        if video_capture2:
            return video_capture2, "capture2"
        if video_capture1:
            return video_capture1, "capture1"
    return "", ""


def assign_split(trial_id: str, val_ratio: float, seed: int) -> str:
    token = f"{trial_id}:{seed}".encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    score = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "val" if score < val_ratio else "train"


def sample_frame_indices(
    frame_count: int,
    frames_per_trial: int,
    *,
    min_frame: int = 0,
    max_frame: int | None = None,
    sampling: str = "uniform",
    rng: np.random.Generator | None = None,
) -> list[int]:
    if frame_count <= 0 or frames_per_trial <= 0:
        return []

    lo = max(0, int(min_frame))
    hi_cap = frame_count - 1
    hi = hi_cap if max_frame is None else min(hi_cap, int(max_frame))
    if hi < lo:
        return []

    available = hi - lo + 1
    k = min(frames_per_trial, available)
    if sampling == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        picks = rng.choice(np.arange(lo, hi + 1, dtype=int), size=k, replace=False)
        return sorted(int(x) for x in picks.tolist())

    # Uniform sampling over the selected range.
    if k == 1:
        return [lo]
    picks = np.linspace(lo, hi, num=k)
    return sorted({int(round(x)) for x in picks.tolist()})


def iter_selected_trials(
    trial_index_df: pd.DataFrame,
    task_filter: set[str] | None,
    max_trials: int | None = None,
) -> Iterable[pd.Series]:
    count = 0
    for _, row in trial_index_df.iterrows():
        task_name = str(row.get("task_name", "")).strip()
        if task_filter and task_name.lower() not in task_filter:
            continue
        yield row
        count += 1
        if max_trials is not None and count >= max_trials:
            break


def prepare_labeling_dataset(
    trial_index_csv: str | Path,
    output_root: str | Path,
    *,
    task_names: list[str] | None = None,
    capture_preference: str = "capture2",
    frames_per_trial: int = 20,
    max_trials: int | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_frame: int = 0,
    max_frame: int | None = None,
    sampling: str = "uniform",
    image_ext: str = "jpg",
    create_empty_labels: bool = True,
) -> pd.DataFrame:
    trial_index_csv = Path(trial_index_csv)
    output_root = Path(output_root)
    images_root = output_root / "images"
    labels_root = output_root / "labels"
    manifests_root = output_root / "manifests"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    trial_df = pd.read_csv(trial_index_csv)
    if trial_df.empty:
        raise ValueError(f"Trial index is empty: {trial_index_csv}")

    normalized_tasks = {name.strip().lower() for name in (task_names or []) if name.strip()}
    task_filter = normalized_tasks if normalized_tasks else None
    sampling = sampling.lower().strip()
    if sampling not in {"uniform", "random"}:
        raise ValueError("sampling must be one of: uniform, random")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for row in iter_selected_trials(trial_df, task_filter, max_trials=max_trials):
        trial_id = str(row.get("trial_id", "")).strip() or "unknown_trial"
        task_name = str(row.get("task_name", "")).strip() or "unknown_task"
        video_path, used_capture = choose_video_path(row, capture_preference=capture_preference)
        if not video_path:
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_indices = sample_frame_indices(
            frame_count,
            frames_per_trial,
            min_frame=min_frame,
            max_frame=max_frame,
            sampling=sampling,
            rng=rng,
        )

        split = assign_split(trial_id=trial_id, val_ratio=val_ratio, seed=seed)
        split_images = images_root / split
        split_labels = labels_root / split
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        safe_task = _sanitize(task_name)
        safe_trial = _sanitize(trial_id)
        safe_capture = _sanitize(used_capture)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            stem = f"{safe_task}_{safe_trial}_{safe_capture}_f{int(frame_idx):06d}"
            image_path = split_images / f"{stem}.{image_ext}"
            label_path = split_labels / f"{stem}.txt"

            cv2.imwrite(str(image_path), frame)
            if create_empty_labels and not label_path.exists():
                label_path.write_text("", encoding="utf-8")

            frame_time = float(frame_idx) / fps if fps > 0 else -1.0
            rows.append(
                {
                    "task_name": task_name,
                    "trial_id": trial_id,
                    "split": split,
                    "capture": used_capture,
                    "video_path": video_path,
                    "frame_index": int(frame_idx),
                    "frame_time_sec": frame_time,
                    "image_path": str(image_path.resolve()),
                    "label_path": str(label_path.resolve()),
                }
            )
        cap.release()

    manifest = pd.DataFrame(rows)
    manifest_path = manifests_root / "labeling_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest
