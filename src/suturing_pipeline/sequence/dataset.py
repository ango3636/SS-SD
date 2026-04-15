from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def resize_and_normalize_bgr(frame: np.ndarray, img_size: int = 224) -> torch.Tensor:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = (frame - mean) / std
    frame = np.transpose(frame, (2, 0, 1))
    return torch.tensor(frame, dtype=torch.float32)


class ClipSequenceDataset(Dataset):
    def __init__(self, clips_df: pd.DataFrame, img_size: int = 224, labels_csv: str | None = None):
        self.clips_df = clips_df.reset_index(drop=True)
        self.img_size = img_size
        self.label_map: dict[str, int] = {}
        if labels_csv is not None and Path(labels_csv).exists():
            labels_df = pd.read_csv(labels_csv)
            self.label_map = dict(zip(labels_df["clip_id"], labels_df["label"]))

    def __len__(self) -> int:
        return len(self.clips_df)

    def __getitem__(self, idx: int):
        row = self.clips_df.iloc[idx]
        clip_dir = Path(row["clip_dir"])
        frame_files = sorted(list(clip_dir.glob("*.jpg")))

        frames = []
        for fp in frame_files:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            frames.append(resize_and_normalize_bgr(img, self.img_size))

        clip_tensor = torch.stack(frames, dim=0)
        label = self.label_map.get(row["clip_id"], -1)
        return {
            "clip": clip_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "clip_id": row["clip_id"],
        }
