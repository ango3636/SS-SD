"""Automatic speech evaluation for narration runs (WER vs intended script).

Uses a small Whisper model for ASR and ``jiwer`` for word error rate. Optional
dependencies are imported lazily so imports of other audio modules stay light.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

_ASR_PIPE: Any = None


def normalize_text_for_wer(text: str) -> str:
    s = (text or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def reference_from_narration_segments(segments: list[dict]) -> str:
    parts: list[str] = []
    for seg in segments:
        t = str(seg.get("narration_text", "")).strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def _resolve_pipeline_device(device: Optional[str]) -> int | str:
    if device:
        d = device.strip().lower()
        if d.startswith("cuda"):
            return 0
        if d == "mps":
            return "mps"
        return -1
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return -1


def _get_whisper_pipe(model_id: str, device: Optional[str]) -> Any:
    global _ASR_PIPE
    if _ASR_PIPE is not None:
        return _ASR_PIPE
    from transformers import pipeline

    dev = _resolve_pipeline_device(device)
    _ASR_PIPE = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=dev,
    )
    return _ASR_PIPE


def transcribe_narration_audio(
    audio_path: str | Path,
    *,
    model_id: str = "openai/whisper-tiny",
    device: Optional[str] = None,
) -> str:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Narration audio not found: {path}")
    pipe = _get_whisper_pipe(model_id, device)
    out = pipe(str(path.resolve()))
    if isinstance(out, dict) and "text" in out:
        return str(out["text"]).strip()
    return str(out).strip()


def compute_word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    ref_n = normalize_text_for_wer(reference)
    hyp_n = normalize_text_for_wer(hypothesis)
    if not ref_n:
        return None
    if not hyp_n:
        return 1.0
    from jiwer import wer

    try:
        return float(wer(ref_n, hyp_n))
    except Exception:
        return None


def compute_speech_eval_block(
    narration_segments: list[dict],
    audio_path: str | Path,
    *,
    device: Optional[str] = None,
    model_id: str = "openai/whisper-tiny",
) -> Dict[str, object]:
    """Return ``eval.speech``-shaped dict for ``metadata.json``."""
    ref = reference_from_narration_segments(narration_segments)
    block: Dict[str, object] = {
        "wer": None,
        "accuracy": None,
        "summary_quality": None,
        "reference_policy": "intended_script_segments",
        "asr_model": model_id,
    }
    hyp = transcribe_narration_audio(
        audio_path, model_id=model_id, device=device
    )
    block["asr_transcript"] = hyp[:2000] if hyp else ""
    w = compute_word_error_rate(ref, hyp)
    block["wer"] = w
    return block
