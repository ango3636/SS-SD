"""Parsing utilities for JIGSAWS dataset files.

Handles metafiles, transcriptions, kinematics, and experimental-setup
split files in the standard JIGSAWS directory layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_metafile(metafile_path: str | Path) -> Dict[str, str]:
    """Parse a JIGSAWS meta-file into a trial-name -> skill-level map.

    The file is tab-delimited with the actual layout::

        Knot_Tying_B001\\tN\\t13\\t2\\t2\\t...

    Column 0 is the trial name and column 1 is the single-letter skill
    code (``N``/``I``/``E``).  This function normalises to the full word.
    """
    skill_map: Dict[str, str] = {}
    _abbrev = {"N": "Novice", "I": "Intermediate", "E": "Expert"}

    for line in Path(metafile_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Try tab-split first (actual JIGSAWS format), fall back to whitespace
        parts = line.split("\t") if "\t" in line else line.split()
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            continue
        trial_name = parts[0]
        raw_skill = parts[1]
        skill_map[trial_name] = _abbrev.get(raw_skill, raw_skill)

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
    """Return trial names whose skill level is *Expert*.

    Handles both single-letter (``E``) and full-word (``Expert``) encodings.
    """
    skill_map = parse_metafile(metafile_path)
    return [
        name
        for name, level in skill_map.items()
        if level.upper().startswith("E")
    ]
