import tempfile
from pathlib import Path

from suturing_pipeline.data.data_utils import (
    filter_expert_trials,
    parse_metafile,
)
from suturing_pipeline.data.jigsaws_metafile_layout import (
    GRS_MODIFIED_ORDER,
    read_metafile_rows,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_read_metafile_rows_parses_all_columns():
    t = Path(tempfile.mkstemp(suffix=".txt")[1])
    try:
        _write(
            t,
            "Suturing_B001\tE\t9.0\t1\t2\t3\t4\t5\t6\n"
            "Suturing_B002 N 5.0 0 0 0 0 0 0\n",
        )
        rows = read_metafile_rows(t)
        assert len(rows) == 2
        a, b = rows[0], rows[1]
        assert a.trial_name == "Suturing_B001"
        assert a.skill_self_proclaimed == "E"
        assert a.skill_grs == "9.0"
        assert a.grs_respect_tissue == "1"
        assert a.grs_suture_needle == "2"
        assert a.grs_time_and_motion == "3"
        assert a.grs_flow == "4"
        assert a.grs_overall_performance == "5"
        assert a.grs_quality_of_final_product == "6"
        assert b.trial_name == "Suturing_B002"
        assert b.skill_self_proclaimed_letter == "N"
    finally:
        t.unlink(missing_ok=True)


def test_parse_metafile_uses_self_column_not_grs():
    t = Path(tempfile.mkstemp(suffix=".txt")[1])
    try:
        # col2 = I, col3 = would-be "E" (GRS) — expert filter must use col2 only
        _write(t, "Suturing_X001\tI\tE\t0\t0\t0\t0\t0\t0\n")
        m = parse_metafile(t)
        assert m["Suturing_X001"] == "Intermediate"
    finally:
        t.unlink(missing_ok=True)


def test_filter_expert_trials_uses_self_column():
    t = Path(tempfile.mkstemp(suffix=".txt")[1])
    try:
        _write(
            t,
            "Suturing_E001\tE\t0\t0\t0\t0\t0\t0\t0\n"
            "Suturing_N001\tN\t0\t0\t0\t0\t0\t0\t0\n",
        )
        experts = filter_expert_trials(t)
        assert experts == ["Suturing_E001"]
    finally:
        t.unlink(missing_ok=True)


def test_grs_order_length():
    assert len(GRS_MODIFIED_ORDER) == 6
