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


def _pick_onetrialout_fold(
    split_type_dir: Path,
    itr: int,
    split: str,
) -> tuple[Path, Path]:
    """Pick the first ``*_Out`` folder (name-sorted) that contains the split file.

    Drive exports and partial zips often omit early folds or leave ``itr_1``
    empty on some folds; the naive ``sorted(...)[0]`` choice then fails even
    when a later fold has ``Train.txt`` / ``Test.txt``.
    """
    if not split_type_dir.is_dir():
        raise FileNotFoundError(
            f"Experimental-setup directory not found: {split_type_dir}"
        )
    outs = sorted(
        (
            p
            for p in split_type_dir.iterdir()
            if p.is_dir() and p.name.endswith("_Out")
        ),
        key=lambda p: p.name,
    )
    if not outs:
        raise FileNotFoundError(
            f"No *_Out folders found under {split_type_dir}"
        )
    for out_dir in outs:
        split_path = _resolve_ci(out_dir / f"itr_{itr}" / f"{split}.txt")
        if split_path.exists():
            return out_dir, split_path

    first = outs[0]
    split_file = _resolve_ci(first / f"itr_{itr}" / f"{split}.txt")
    avail_out = [p.name for p in outs]
    itr_dir = _resolve_ci(first / f"itr_{itr}")
    itr_hint = ""
    if itr_dir.is_dir():
        itr_hint = f" Example {itr_dir} contents: {sorted(p.name for p in itr_dir.iterdir())!r}."
    raise FileNotFoundError(
        f"No {split}.txt (case-insensitive) found under itr_{itr} in any *_Out folder "
        f"under {split_type_dir}. Folds present: {avail_out[:30]}"
        f"{'...' if len(avail_out) > 30 else ''}.{itr_hint}\n"
        f"Re-sync the full Experimental_setup tree from JIGSAWS, or pass "
        f"--held_out <N> for a fold that contains Train.txt."
    )


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
        If *True*, keep only trials whose **self-reported** skill in the meta
        file (column 2, 1-based; see ``jigsaws_metafile_layout``) is *Expert*
        (``E``), not the GRS column.
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
            split_file = _resolve_ci(out_dir / f"itr_{itr}" / f"{split}.txt")
            if not split_file.exists():
                itr_dir = _resolve_ci(out_dir / f"itr_{itr}")
                itr_hint = ""
                if itr_dir.is_dir():
                    itr_hint = (
                        f" Contents of {itr_dir}: "
                        f"{sorted(p.name for p in itr_dir.iterdir())!r}."
                    )
                raise FileNotFoundError(
                    f"Split file not found: {split_file}\n"
                    f"Expected JIGSAWS experimental-setup lists under "
                    f"Experimental_setup/Suturing/<Balanced|unBalanced>/"
                    f"GestureClassification/<split_type>/*_Out/itr_<n>/Train.txt "
                    f"(filename case-insensitive).{itr_hint}\n"
                    f"Try another fold, e.g. --held_out 10 --itr 1."
                )
        else:
            out_dir, split_file = _pick_onetrialout_fold(split_type_dir, itr, split)

        task_dir = _resolve_ci(self.data_root / _TASK_NAME)

        # JIGSAWS metafiles are named "meta_file_{TaskName}.txt", not
        # "metafile.txt".  Search for the actual file.  Column layout (1-based:
        # filename, self N/I/E, GRS, six GRS sub-scores) is documented in
        # ``suturing_pipeline.data.jigsaws_metafile_layout``; *expert_only* uses
        # column 2 (self-reported), not the GRS column.
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
