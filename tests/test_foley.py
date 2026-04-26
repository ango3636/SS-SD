"""Tests for gesture-keyed WAV foley helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from suturing_pipeline.audio.foley import (
    FoleyLibrary,
    place_foley_on_segment,
)


def test_place_foley_start_and_center() -> None:
    f = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    s = place_foley_on_segment(f, 6, align="start")
    assert s.shape == (6,)
    assert np.allclose(s[:3], 1.0)
    assert np.allclose(s[3:], 0.0)
    c = place_foley_on_segment(f, 7, align="center")
    assert c.shape == (7,)
    assert float(np.sum(c > 0)) == 3.0


def test_foley_library_loads_and_resolves(tmp_path: Path) -> None:
    sr = 8000
    wav_path = tmp_path / "G3.wav"
    pcm = (np.random.randint(-1000, 1000, size=400, dtype=np.int16))
    wavfile.write(str(wav_path), sr, pcm)

    lib = FoleyLibrary(tmp_path, sample_rate=24000)
    assert lib.enabled
    mono = lib.get_mono("G3")
    assert mono is not None
    assert mono.ndim == 1
    assert mono.shape[0] == int(round(400 * (24000 / sr)))
    assert lib.get_mono("G99") is None


def test_foley_library_case_insensitive_stem(tmp_path: Path) -> None:
    p = tmp_path / "g2.WAV"
    wavfile.write(str(p), 24000, np.zeros(80, dtype=np.int16))
    lib = FoleyLibrary(tmp_path, sample_rate=24000)
    assert lib.get_mono("G2") is not None
