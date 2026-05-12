"""Parsing utilities for JIGSAWS dataset files.

Handles metafiles, transcriptions, kinematics, and experimental-setup
split files in the standard JIGSAWS directory layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from suturing_pipeline.data.jigsaws_metafile_layout import (
    GRS_MODIFIED_ORDER,
    JigsawsMetafileRow,
    read_metafile_rows,
)

_MAP_SELF_SKILL = {"N": "Novice", "I": "Intermediate", "E": "Expert"}


def _self_skill_to_label(raw: str) -> str:
    """Map column-2 (1-based) self-reported code to a normalised label."""
    s = (raw or "").strip()
    if not s:
        return ""
    key = s[:1].upper()
    if key in _MAP_SELF_SKILL:
        return _MAP_SELF_SKILL[key]
    return s


def parse_metafile(metafile_path: str | Path) -> Dict[str, str]:
    """Parse a JIGSAWS meta file into a trial name → self-reported skill *label* map.

    The meta file (e.g. ``meta_file_Suturing.txt``) is tab- or
    whitespace-delimited. The official JIGSAWS **1-based** column layout is
    described in :mod:`~suturing_pipeline.data.jigsaws_metafile_layout` and
    in :class:`JigsawsMetafileRow` — in short:

    * **1** — filename / trial name
    * **2** — self-reported skill: ``N``/``I``/``E`` (novice / intermediate / expert)
    * **3** — GRS (global rating scale) skill
    * **4–9** — six modified GRS element scores, in
      :data:`~suturing_pipeline.data.jigsaws_metafile_layout.GRS_MODIFIED_ORDER`

    This function only uses **columns 1 and 2** to build
    ``{ trial_name: "Novice" | "Intermediate" | "Expert" }`` from the
    **self-reported** column. It does **not** use column 3 (GRS) for this map.

    For full rows including GRS fields, use :func:`read_metafile_rows`.
    """
    skill_map: Dict[str, str] = {}
    for row in read_metafile_rows(metafile_path):
        skill_map[row.trial_name] = _self_skill_to_label(row.skill_self_proclaimed)
    return skill_map


def parse_transcription(
    transcription_path: str | Path,
) -> List[Tuple[int, int, str]]:
    """Parse a JIGSAWS gesture-transcription file.

    Each line has the format ``START_FRAME END_FRAME GESTURE_LABEL``.

    Returns a list of ``(start_frame, end_frame, gesture_label)`` tuples.
    """
    segments: List[Tuple[int, int, str]] = []
    for line in Path(transcription_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        segments.append((int(parts[0]), int(parts[1]), parts[2]))
    return segments


def parse_kinematics(kinematics_path: str | Path) -> np.ndarray:
    """Load a JIGSAWS kinematics file as a NumPy array.

    The file is headerless, whitespace-delimited, with 76 columns per row
    (four 19-column tool blocks: master left/right, slave left/right — each
    with tip xyz, rotation ``R``, translational velocity, rotational velocity,
    and gripper angle). See :mod:`~suturing_pipeline.data.jigsaws_kinematics_layout`.

    Returns an array of shape ``(num_frames, 76)``.
    """
    data = np.loadtxt(str(kinematics_path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_split(split_txt_path: str | Path) -> List[str]:
    """Read a JIGSAWS ``Train.txt`` or ``Test.txt`` split file and return
    the unique trial names referenced therein.

    JIGSAWS split files list gesture *segments*, not plain trial names.
    Each line looks like::

        Knot_Tying_D001_000041_000170.txt   G1

    This function extracts the trial name (``Knot_Tying_D001``) from each
    entry and returns a deduplicated, sorted list.
    """
    import re

    trial_set: set[str] = set()
    for line in Path(split_txt_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # First token is the segment filename
        token = line.split()[0]
        # Strip .txt suffix if present
        token = token.removesuffix(".txt")
        # Pattern: {TaskName}_{TrialSuffix}_{StartFrame}_{EndFrame}
        # Trial suffix is a letter + digits (e.g. B001, D001).
        # We greedily match up to the trial suffix, leaving the
        # numeric-only start/end frames at the end.
        m = re.match(r"^(.+?_[A-Za-z]\d+)_\d+_\d+$", token)
        if m:
            trial_set.add(m.group(1))
        else:
            # Fallback: maybe it *is* a plain trial name
            trial_set.add(token)
    return sorted(trial_set)


def get_frame_label_map(
    transcription: List[Tuple[int, int, str]],
) -> Dict[int, str]:
    """Expand transcription segments into a per-frame gesture-label mapping.

    Both ``start_frame`` and ``end_frame`` are treated as inclusive.
    """
    frame_map: Dict[int, str] = {}
    for start, end, label in transcription:
        for f in range(start, end + 1):
            frame_map[f] = label
    return frame_map


def filter_expert_trials(metafile_path: str | Path) -> List[str]:
    """Return trial names where **self-reported** skill (meta file **column 2**,
    1-based) is *Expert*.

    Uses the same N/I/E field as :func:`parse_metafile`, **not** column 3
    (GRS). Expert is detected when the normalised label from column 2 starts
    with ``"E"`` (see :func:`parse_metafile`).

    See :mod:`suturing_pipeline.data.jigsaws_metafile_layout` for the full
    column specification.
    """
    skill_map = parse_metafile(metafile_path)
    return [
        name
        for name, level in skill_map.items()
        if level.upper().startswith("E")
    ]
