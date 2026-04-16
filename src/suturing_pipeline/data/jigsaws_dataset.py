"""PyTorch Dataset for JIGSAWS kinematic-conditioned frame generation.

Lazily extracts video frames via OpenCV so that the full video is never
loaded into memory.  Kinematics are normalised with a
:class:`~sklearn.preprocessing.StandardScaler` fitted at ``__init__`` time
and exposed via :pyattr:`scaler` / :pyattr:`gesture_to_int` for
serialisation alongside model checkpoints.

Only the JIGSAWS ``suturing`` task is supported — kinematics for the
other JIGSAWS tasks are not available in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .data_utils import (
    filter_expert_trials,
    get_frame_label_map,
    load_split,
    parse_kinematics,
    parse_transcription,
)

# Only the suturing task is supported.  JIGSAWS directory names use
# lowercase (``suturing/``) while filenames use CamelCase (``Suturing_B001``);
# these constants bridge the two conventions.
_TASK_NAME = "suturing"
_TASK_FILENAME_PREFIX = "Suturing"


def _resolve_ci(path: Path) -> Path:
    """Resolve *path* with case-insensitive segment matching.

    Walks the path from root to leaf. At each level, if the exact segment
    does not exist on disk, a case-insensitive scan of the parent directory
    is performed. This handles the JIGSAWS Google-Drive convention where
    folder names use mixed CamelCase (``Experimental_setup``,
    ``GestureClassification``, ``Train.txt``, etc.) while the code uses
    lowercase for portability.
    """
    parts = path.parts
    resolved = Path(parts[0])
    for segment in parts[1:]:
        candidate = resolved / segment
        if candidate.exists():
            resolved = candidate
            continue
        # case-insensitive fallback
        seg_lower = segment.lower()
        try:
            matches = [
                p for p in resolved.iterdir() if p.name.lower() == seg_lower
            ]
        except (OSError, PermissionError):
            return path  # give up, return as-is
        if matches:
            resolved = matches[0]
        else:
            return path
    return resolved


class JIGSAWSDataset(Dataset):
    """Frame-level dataset that yields ``(frame, kinematics, gesture_label)``
    triplets aligned by frame index.

    Parameters
    ----------
    data_root:
        Root of the JIGSAWS download (must contain ``suturing/``).
    split:
        ``"train"`` or ``"test"``.
    split_type:
        Experimental-setup split strategy (default ``"onetrialout"``).
    balance:
        ``"balanced"`` or ``"unbalanced"`` (default ``"balanced"``).
    held_out:
        Which trial group to hold out (e.g. ``10`` for ``10_Out``).
        In the standard JIGSAWS OneTrialOut setup, each ``{N}_Out``
        folder represents a leave-one-out fold.  If *None*, the first
        available ``*_Out`` folder is used automatically.
    itr:
        Cross-validation iteration number (default ``1``).
    expert_only:
        If *True*, keep only trials whose metafile skill level is *Expert*.
    modality:
        ``"video"``, ``"kinematics"``, or ``"both"`` (default ``"both"``).
    image_size:
        Spatial size to which video frames are resized (default ``256``).
    capture:
        Which camera view to use — ``1`` or ``2`` (default ``1``).
    frame_stride:
        Sample every *N*-th frame instead of every frame (default ``1``).
        Set to e.g. ``30`` to sample ~1 frame/second at 30 fps, which
        dramatically reduces dataset size for faster iteration on CPU.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        split_type: str = "onetrialout",
        balance: str = "balanced",
        held_out: int | None = None,
        itr: int = 1,
        expert_only: bool = False,
        modality: str = "both",
        image_size: int = 256,
        capture: int = 1,
        frame_stride: int = 1,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.task = _TASK_NAME
        self.split = split
        self.modality = modality
        self.image_size = image_size
        self.capture = capture

        prefix = _TASK_FILENAME_PREFIX

        # --- resolve paths (case-insensitive for cross-platform compat) ------
        split_type_dir = _resolve_ci(
            self.data_root
            / "experimental_setup"
            / _TASK_NAME
            / balance
            / "gestureclassification"
            / split_type
        )

        # JIGSAWS OneTrialOut has {N}_Out subfolders, not a single "one_out".
        if held_out is not None:
            out_dir = _resolve_ci(split_type_dir / f"{held_out}_Out")
        else:
            # Auto-pick the first available *_Out folder.
            candidates = sorted(
                (p for p in split_type_dir.iterdir() if p.is_dir()),
                key=lambda p: p.name,
            ) if split_type_dir.is_dir() else []
            if not candidates:
                raise FileNotFoundError(
                    f"No held-out folders found under {split_type_dir}"
                )
            out_dir = candidates[0]

        split_file = _resolve_ci(out_dir / f"itr_{itr}" / f"{split}.txt")
        task_dir = _resolve_ci(self.data_root / _TASK_NAME)

        # JIGSAWS metafiles are named "meta_file_{TaskName}.txt", not
        # "metafile.txt".  Search for the actual file.
        metafile = _resolve_ci(task_dir / f"meta_file_{prefix}.txt")
        if not metafile.exists():
            metafile = _resolve_ci(task_dir / "metafile.txt")

        # --- load trial list --------------------------------------------------
        trial_names = load_split(split_file)
        if expert_only and metafile.exists():
            experts = set(filter_expert_trials(metafile))
            trial_names = [t for t in trial_names if t in experts]
        if not trial_names:
            raise RuntimeError(
                f"No trials found for task={_TASK_NAME!r}, split={split!r}, "
                f"expert_only={expert_only}"
            )

        # --- per-trial data ---------------------------------------------------
        self._trial_names: List[str] = []
        self._video_paths: List[Path] = []
        self._kinematics: List[np.ndarray] = []
        self._frame_labels: List[Dict[int, str]] = []
        self._frame_counts: List[int] = []

        all_gesture_labels: set[str] = set()

        skipped: List[Tuple[str, str]] = []
        for trial_name in trial_names:
            suffix = trial_name.replace(prefix + "_", "")
            video_path = _resolve_ci(
                task_dir / "video" / f"{prefix}_{suffix}_capture{capture}.avi"
            )
            kin_path = _resolve_ci(
                task_dir / "kinematics" / "allgestures" / f"{prefix}_{suffix}.txt"
            )
            trans_path = _resolve_ci(
                task_dir / "transcriptions" / f"{prefix}_{suffix}.txt"
            )

            if not video_path.exists():
                skipped.append((trial_name, f"video not found: {video_path}"))
                continue
            if not kin_path.exists():
                skipped.append((trial_name, f"kinematics not found: {kin_path}"))
                continue

            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0:
                continue

            kin = parse_kinematics(kin_path)

            if trans_path.exists():
                transcription = parse_transcription(trans_path)
                frame_label_map = get_frame_label_map(transcription)
            else:
                frame_label_map = {}

            all_gesture_labels.update(frame_label_map.values())

            self._trial_names.append(trial_name)
            self._video_paths.append(video_path)
            self._kinematics.append(kin)
            self._frame_labels.append(frame_label_map)
            self._frame_counts.append(min(frame_count, len(kin)))

        if not self._trial_names:
            detail = "\n".join(f"  {t}: {reason}" for t, reason in skipped[:10])
            raise RuntimeError(
                f"All {len(trial_names)} trials were skipped (missing files).\n"
                f"First skipped trials:\n{detail}"
            )

        # --- flat index -------------------------------------------------------
        stride = max(1, frame_stride)
        self._index: List[Tuple[int, int]] = []
        for trial_idx, fc in enumerate(self._frame_counts):
            for frame_idx in range(0, fc, stride):
                self._index.append((trial_idx, frame_idx))

        # --- kinematics scaler ------------------------------------------------
        self.scaler = StandardScaler()
        all_kin = np.concatenate(self._kinematics, axis=0)
        self.scaler.fit(all_kin)

        # --- gesture label -> int mapping (sorted for determinism) ------------
        sorted_labels = sorted(all_gesture_labels)
        self.gesture_to_int: Dict[str, int] = {
            label: idx for idx, label in enumerate(sorted_labels)
        }
        self.num_gestures = max(len(self.gesture_to_int), 1)

        # Re-usable VideoCapture handles are NOT cached here because Dataset
        # instances may be used across DataLoader workers (each must open its
        # own file descriptor).

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        trial_idx, frame_idx = self._index[idx]

        # --- video frame (lazy seek) -----------------------------------------
        frame_tensor = torch.zeros(3, self.image_size, self.image_size)
        if self.modality in ("video", "both"):
            cap = cv2.VideoCapture(str(self._video_paths[trial_idx]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.image_size, self.image_size))
                frame_tensor = (
                    torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0
                )

        # --- kinematics -------------------------------------------------------
        kin_row = self._kinematics[trial_idx][frame_idx]
        kin_scaled = self.scaler.transform(kin_row.reshape(1, -1)).flatten()
        kin_tensor = torch.from_numpy(kin_scaled).float()

        # --- gesture label ----------------------------------------------------
        label_str = self._frame_labels[trial_idx].get(
            frame_idx, next(iter(self.gesture_to_int), "G0")
        )
        gesture_int = self.gesture_to_int.get(label_str, 0)

        return frame_tensor, kin_tensor, gesture_int
