"""JIGSAWS ``meta_file_{TaskName}.txt`` — column reference and row parsing.

Rows are tab- or whitespace-delimited. **Column numbers below are 1-based** (as in
dataset READMEs). Indexing in code uses 0-based ``parts[i]`` where
``parts[0]`` = column 1, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

PathLike = Union[str, Path]

# 1-based column indices (for documentation and cross-referencing READMEs)
METAFILE_COL_FILENAME_1 = 1
METAFILE_COL_SKILL_SELF_PROCLAIMED_1 = 2
METAFILE_COL_SKILL_GRS_1 = 3
# Columns 4–9 (1-based), six elements of the modified GRS
METAFILE_COL_GRS_RESPECT_TISSUE_1 = 4
METAFILE_COL_GRS_SUTURE_NEEDLE_1 = 5
METAFILE_COL_GRS_TIME_AND_MOTION_1 = 6
METAFILE_COL_GRS_FLOW_1 = 7
METAFILE_COL_GRS_OVERALL_PERFORMANCE_1 = 8
METAFILE_COL_GRS_QUALITY_OF_FINAL_PRODUCT_1 = 9

# Human-readable order for the six GRS sub-scores (columns 4–9, 1-based)
GRS_MODIFIED_ORDER: tuple[str, ...] = (
    "Respect for tissue",
    "Suture/needle handling",
    "Time and motion",
    "Flow of operation",
    "Overall performance",
    "Quality of final product",
)


@dataclass(frozen=True, slots=True)
class JigsawsMetafileRow:
    """One parsed line from a JIGSAWS meta file.

    Fields correspond to 1-based columns 1..9. Missing optional columns
    (e.g. a short or legacy line) are stored as empty strings.
    """

    trial_name: str
    """Column 1 — trial / filename (e.g. ``Suturing_B001``)."""
    skill_self_proclaimed: str
    """Column 2 — self-reported skill (``N`` / ``I`` / ``E``). See also ``skill_self_proclaimed_letter``."""
    skill_grs: str
    """Column 3 — GRS (global rating scale) skill level / score, as in the file."""
    grs_respect_tissue: str
    grs_suture_needle: str
    grs_time_and_motion: str
    grs_flow: str
    grs_overall_performance: str
    grs_quality_of_final_product: str

    @property
    def skill_self_proclaimed_letter(self) -> str:
        """``N`` / ``I`` / ``E`` for display; first letter of column 2."""
        s = (self.skill_self_proclaimed or "").strip().upper()
        return s[:1] if s else "?"


def _split_metafile_line(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if "\t" in line:
        return [p.strip() for p in line.split("\t") if p.strip()]
    return [p for p in line.split() if p.strip()]


def _p(parts: Sequence[str], i: int) -> str:
    return str(parts[i]).strip() if i < len(parts) else ""


def read_metafile_rows(metafile_path: PathLike) -> List[JigsawsMetafileRow]:
    """Parse every data row from a JIGSAWS meta file into :class:`JigsawsMetafileRow`.

    Skips empty lines. Lines with fewer than two fields are ignored.
    """
    path = Path(metafile_path)
    out: List[JigsawsMetafileRow] = []
    for line in path.read_text().splitlines():
        parts = _split_metafile_line(line)
        if len(parts) < 2:
            continue
        out.append(
            JigsawsMetafileRow(
                trial_name=parts[0],
                skill_self_proclaimed=parts[1],
                skill_grs=_p(parts, 2),
                grs_respect_tissue=_p(parts, 3),
                grs_suture_needle=_p(parts, 4),
                grs_time_and_motion=_p(parts, 5),
                grs_flow=_p(parts, 6),
                grs_overall_performance=_p(parts, 7),
                grs_quality_of_final_product=_p(parts, 8),
            )
        )
    return out
