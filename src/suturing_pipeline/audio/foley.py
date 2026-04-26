"""Optional gesture-keyed WAV foley mixed under Bark (and optional AudioGen bed)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

FoleyAlign = Literal["start", "center"]


def _resample_mono(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or w.size == 0:
        return w
    new_len = max(1, int(round(w.size * float(target_sr) / float(orig_sr))))
    x_old = np.linspace(0.0, 1.0, num=w.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, w.astype(np.float64)).astype(np.float32)


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    sr = int(sr)
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        maxv = float(np.iinfo(data.dtype).max)
        x = (x / maxv).astype(np.float32)
    else:
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = (x / peak).astype(np.float32)
    return x, sr


def place_foley_on_segment(
    foley: np.ndarray,
    segment_len: int,
    *,
    align: FoleyAlign = "start",
) -> np.ndarray:
    """Pad or trim ``foley`` to ``segment_len`` samples; ``align`` sets horizontal position."""
    f = np.asarray(foley, dtype=np.float32).reshape(-1)
    if segment_len <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(segment_len, dtype=np.float32)
    if f.size == 0:
        return out
    n = min(f.size, segment_len)
    if align == "start":
        out[:n] = f[:n]
    else:
        off = max(0, (segment_len - f.size) // 2)
        out[off : off + n] = f[:n]
    return out


class FoleyLibrary:
    """Load ``{gesture}.wav`` files from a directory (e.g. ``G3.wav``)."""

    def __init__(self, foley_dir: str | Path, sample_rate: int) -> None:
        self.root = Path(foley_dir).expanduser().resolve()
        self.sample_rate = int(sample_rate)
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return self.root.is_dir()

    def _resolve_path(self, gesture: str) -> Optional[Path]:
        if not self.enabled:
            return None
        g = str(gesture).strip()
        if not g:
            return None
        for name in (f"{g}.wav", f"{g.upper()}.wav", f"{g.lower()}.wav"):
            p = self.root / name
            if p.is_file():
                return p
        try:
            for p in self.root.iterdir():
                if p.is_file() and p.suffix.lower() == ".wav":
                    if p.stem.upper() == g.upper():
                        return p
        except OSError:
            return None
        return None

    def get_mono(self, gesture: str) -> Optional[np.ndarray]:
        key = str(gesture).strip()
        if not key:
            return None
        if key in self._cache:
            c = self._cache[key]
            return None if c.size == 0 else c.copy()
        path = self._resolve_path(gesture)
        if path is None:
            self._cache[key] = np.zeros(0, dtype=np.float32)
            return None
        wav, sr = _read_wav_mono(path)
        wav = _resample_mono(wav, sr, self.sample_rate)
        self._cache[key] = wav
        return wav.copy()


__all__ = [
    "FoleyAlign",
    "FoleyLibrary",
    "place_foley_on_segment",
]
