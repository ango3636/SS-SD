"""Tests for gesture-keyed WAV foley helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from suturing_pipeline.audio.foley import (
    FoleyLibrary,
    place_foley_on_segment,
)
from suturing_pipeline.audio.tts_converter import _foley_layer_waveforms_for_segment


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


def test_foley_library_lr_pair(tmp_path: Path) -> None:
    sr = 8000
    for name, data in (("G2_L.wav", 100), ("G2_R.wav", 120)):
        wavfile.write(
            str(tmp_path / name),
            sr,
            (np.ones(data, dtype=np.float32) * 0.2 * 32767).astype(np.int16),
        )
    lib = FoleyLibrary(tmp_path, sample_rate=8000)
    pair = lib.get_mono_lr_pair("G2")
    assert pair is not None
    assert pair[0].shape[0] == 100 and pair[1].shape[0] == 120


def test_foley_library_lr_pair_requires_both(tmp_path: Path) -> None:
    wavfile.write(str(tmp_path / "G2_L.wav"), 8000, np.zeros(50, dtype=np.int16))
    lib = FoleyLibrary(tmp_path, sample_rate=8000)
    assert lib.get_mono_lr_pair("G2") is None


def test_foley_layer_waveforms_priority(tmp_path: Path) -> None:
    sr = 8000
    wavfile.write(str(tmp_path / "G1.wav"), sr, np.ones(30, dtype=np.int16))
    wavfile.write(str(tmp_path / "G2.wav"), sr, np.ones(40, dtype=np.int16) * 2)
    lib = FoleyLibrary(tmp_path, sample_rate=8000)
    seg = {"gesture": "G9", "foley_gestures": ["G1", "G2"]}
    layers = _foley_layer_waveforms_for_segment(seg, lib)
    assert len(layers) == 2
    assert layers[0].shape[0] == 30 and layers[1].shape[0] == 40


def test_foley_layer_waveforms_left_right_keys(tmp_path: Path) -> None:
    sr = 8000
    wavfile.write(str(tmp_path / "G3.wav"), sr, np.ones(20, dtype=np.int16))
    wavfile.write(str(tmp_path / "G4.wav"), sr, np.ones(25, dtype=np.int16) * 3)
    lib = FoleyLibrary(tmp_path, sample_rate=8000)
    seg = {"gesture": "G1", "foley_gesture_right": "G4", "foley_gesture_left": "G3"}
    layers = _foley_layer_waveforms_for_segment(seg, lib)
    assert len(layers) == 2


def test_foley_layer_lr_files(tmp_path: Path) -> None:
    sr = 8000
    wavfile.write(str(tmp_path / "GX_L.wav"), sr, np.ones(10, dtype=np.int16))
    wavfile.write(str(tmp_path / "GX_R.wav"), sr, np.ones(12, dtype=np.int16))
    lib = FoleyLibrary(tmp_path, sample_rate=8000)
    seg = {"gesture": "GX"}
    layers = _foley_layer_waveforms_for_segment(seg, lib)
    assert len(layers) == 2
