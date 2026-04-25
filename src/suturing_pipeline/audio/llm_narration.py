"""Free / local LLM backends for gesture narration (text before Bark TTS).

Primary path: **Ollama** (https://ollama.com) — no cloud bill; run a model locally.

Optional: **Hugging Face Inference API** with a read token (free tier limits apply).

Flow is always speech-first: build prompt → LLM returns one line → sanitize →
Bark consumes ``narration_text``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, Mapping, Optional, cast

from suturing_pipeline.audio.narration_templates import (
    build_llm_prompt,
    max_narration_words_for_duration,
    render_narration_text,
)

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
# Small instruct model often available on HF free inference (swap if your token lacks access).
DEFAULT_HF_NARRATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def _sanitize_narration_line(raw: str, max_words: int) -> str:
    """Take first line, strip quotes, enforce word cap for TTS."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("LLM returned empty narration.")
    s = s.split("\n")[0].strip()
    s = re.sub(r"^[\"']|[\"']$", "", s).strip()
    s = re.sub(r"^(narration|text|output)\s*:\s*", "", s, flags=re.I).strip()
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s


def _http_json_post(url: str, body: dict, headers: dict, timeout_sec: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def complete_narration_ollama(
    *,
    user_prompt: str,
    base_url: str = DEFAULT_OLLAMA_BASE,
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout_sec: float = 120.0,
) -> str:
    """Chat completion via local Ollama ``/api/chat``."""
    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "stream": False,
    }
    data = _http_json_post(
        url, body, headers={"Content-Type": "application/json"}, timeout_sec=timeout_sec
    )
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Unexpected Ollama response: {data!r}")
    return content


def complete_narration_huggingface(
    *,
    user_prompt: str,
    model_id: str,
    token: str,
    timeout_sec: float = 120.0,
) -> str:
    """Text generation via Hugging Face Serverless Inference API."""
    # Classic inference endpoint (works for many causal LMs with a single prompt string).
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    body = {
        "inputs": user_prompt,
        "parameters": {
            "max_new_tokens": 96,
            "return_full_text": False,
            "temperature": 0.5,
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        data = _http_json_post(url, body, headers=headers, timeout_sec=timeout_sec)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"HuggingFace inference HTTP {e.code} for model {model_id!r}. {detail} "
            "Check HF_TOKEN access to this model, or use --narration_backend ollama."
        ) from e

    # Response shapes vary: [{"generated_text": "..."}] or {"generated_text": "..."}
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        text = data.get("generated_text")
        if isinstance(text, str) and text.strip():
            return text
    raise RuntimeError(f"Unexpected HuggingFace response shape: {data!r}")


def synthesize_narration_line(
    *,
    gesture_label: str,
    gesture_description: str,
    kinematic_summary: Mapping[str, object],
    duration_seconds: float,
    backend: str,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    hf_model: str = DEFAULT_HF_NARRATION_MODEL,
    hf_token: Optional[str] = None,
    timeout_sec: float = 120.0,
) -> str:
    """Return one narration line from the chosen backend (speech text only)."""
    b = (backend or "template").strip().lower()
    max_words = max_narration_words_for_duration(duration_seconds)
    user_prompt = build_llm_prompt(
        gesture_label=gesture_label,
        gesture_description=gesture_description,
        kinematic_summary=dict(kinematic_summary),
        duration_seconds=duration_seconds,
    )
    if b == "template":
        return render_narration_text(
            gesture_label, cast(Dict[str, float | str], dict(kinematic_summary))
        )
    if b == "ollama":
        raw = complete_narration_ollama(
            user_prompt=user_prompt,
            base_url=ollama_base_url,
            model=ollama_model,
            timeout_sec=timeout_sec,
        )
        return _sanitize_narration_line(raw, max_words)
    if b in ("huggingface", "hf"):
        tok = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
        if not tok:
            raise RuntimeError(
                "HuggingFace narration requires HF_TOKEN in the environment "
                "or hf_token=... . Get a free token at https://huggingface.co/settings/tokens"
            )
        raw = complete_narration_huggingface(
            user_prompt=user_prompt,
            model_id=hf_model,
            token=tok,
            timeout_sec=timeout_sec,
        )
        return _sanitize_narration_line(raw, max_words)
    raise ValueError(f"Unknown narration backend: {backend!r}")


def apply_llm_narration_to_segments(
    segments: list[Dict[str, object]],
    *,
    backend: str = "template",
    ollama_base_url: str = DEFAULT_OLLAMA_BASE,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    hf_model: str = DEFAULT_HF_NARRATION_MODEL,
    hf_token: Optional[str] = None,
    timeout_sec: float = 120.0,
) -> list[Dict[str, object]]:
    """Return a copy of segments with ``narration_text`` replaced when backend != template."""
    if (backend or "template").lower() == "template":
        return list(segments)
    out: list[Dict[str, object]] = []
    for seg in segments:
        start = float(seg["start_time"])
        end = float(seg["end_time"])
        duration = max(1e-3, end - start)
        summary = seg.get("summary")
        if not isinstance(summary, dict):
            raise TypeError("segment missing kinematic summary dict")
        text = synthesize_narration_line(
            gesture_label=str(seg.get("gesture", "G1")),
            gesture_description=str(
                seg.get("gesture_description")
                or seg.get("gesture", "gesture")
            ),
            kinematic_summary=summary,
            duration_seconds=duration,
            backend=backend,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            hf_model=hf_model,
            hf_token=hf_token,
            timeout_sec=timeout_sec,
        )
        new = dict(seg)
        new["narration_text"] = text
        out.append(new)
    return out


__all__ = [
    "DEFAULT_HF_NARRATION_MODEL",
    "DEFAULT_OLLAMA_BASE",
    "DEFAULT_OLLAMA_MODEL",
    "apply_llm_narration_to_segments",
    "complete_narration_huggingface",
    "complete_narration_ollama",
    "synthesize_narration_line",
]
