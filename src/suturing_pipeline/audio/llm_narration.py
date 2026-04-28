"""Free / local LLM backends for gesture narration (text before Bark TTS).

Primary path: **Ollama** (https://ollama.com) — no cloud bill; run a model on the
**same machine as Python** (default ``http://127.0.0.1:11434``).

Optional: **Hugging Face** router (``router.huggingface.co/v1/chat/completions``) with
a read token (free tier limits apply). Default Llama weights are **gated** — accept the
license on the Hub for your account, or set ``--hf_narration_model`` to another model from
https://huggingface.co/inference/models .

**Google Colab / remote GPU VMs:** ``127.0.0.1`` is the notebook server, not your
laptop. A Cloudflare (or ngrok, etc.) tunnel exposes a *web* URL to your browser;
it does **not** make ``urllib`` calls from Colab reach Ollama on your home PC.
Use ``--narration_backend huggingface`` with ``HF_TOKEN``, install Ollama *inside*
the VM, or point ``--ollama_base_url`` at an Ollama instance that is actually
reachable from that VM (advanced).

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
# Routed via HF ``/v1/chat/completions`` (not hf-inference-only). Gated models require
# accepting the license on the Hub. Override with --hf_narration_model if needed.
DEFAULT_HF_NARRATION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


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


def _is_connection_refused(exc: BaseException) -> bool:
    s = str(exc).lower()
    if "connection refused" in s or "[errno 111]" in s or "errno 111" in s:
        return True
    r = getattr(exc, "reason", None)
    if isinstance(r, OSError) and r.errno == 111:
        return True
    if isinstance(r, ConnectionRefusedError):
        return True
    return False


def _http_json_post(url: str, body: dict, headers: dict, timeout_sec: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            pass
        hint = ""
        if e.code == 404:
            hint = (
                " Pick a model served for chat/inference on the Hub, e.g. "
                "https://huggingface.co/inference/models — or pass a different "
                "--hf_narration_model."
            )
        elif e.code == 400:
            hint = (
                " Often means the model is not available on the chosen route, or the "
                "request body does not match the API. Try another --hf_narration_model "
                "from https://huggingface.co/inference/models"
            )
        elif e.code == 403:
            hint = (
                " Gated model: open the model page on huggingface.co, accept the license, "
                "and ensure your HF_TOKEN belongs to that same account."
            )
        raise RuntimeError(
            f"HTTP {e.code} from inference API ({url!r}). {detail} {hint}".strip()
        ) from e
    except urllib.error.URLError as e:
        hint = ""
        if _is_connection_refused(e):
            if "127.0.0.1" in url or "localhost" in url:
                hint = (
                    " Nothing is listening on this host/port. For Ollama: start the "
                    "daemon on *this same machine* (`ollama serve` or the Ollama app), "
                    "then `ollama pull <model>` (e.g. llama3.2). "
                    "If this code runs on a remote server or Colab, 127.0.0.1 refers to "
                    "that remote box—not your laptop; install/run Ollama there, use an "
                    "SSH tunnel, or switch to --narration_backend huggingface with HF_TOKEN."
                )
            elif "huggingface" in url.lower():
                hint = (
                    " Outbound HTTPS to Hugging Face was refused or blocked. "
                    "Try another network/VPN, or confirm no proxy sends inference traffic to localhost."
                )
        raise RuntimeError(
            f"LLM HTTP request failed for {url!r}: {e!s}.{hint}"
        ) from e
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
    """Chat completion via Hugging Face router (OpenAI-compatible ``/v1/chat/completions``).

    Uses the unified router so HF can pick a working provider. The legacy
    ``/hf-inference/models/{id}`` + ``inputs`` payload only supports models
    registered on ``hf-inference`` and often returns 400 "Model not supported".
    """
    url = "https://router.huggingface.co/v1/chat/completions"
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 150,
        "temperature": 0.5,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = _http_json_post(url, body, headers=headers, timeout_sec=timeout_sec)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise RuntimeError(f"Unexpected HuggingFace chat response shape: {data!r}")


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
) -> tuple[str, Optional[str]]:
    """Return ``(narration_text, llm_user_prompt)``.

    For ``template`` backend the prompt tuple element is ``None`` (no LLM).
    For ``ollama`` / ``huggingface`` the second value is the exact user message
    sent to the chat API (from :func:`build_llm_prompt`).
    """
    b = (backend or "template").strip().lower()
    max_words = max_narration_words_for_duration(duration_seconds)
    user_prompt = build_llm_prompt(
        gesture_label=gesture_label,
        gesture_description=gesture_description,
        kinematic_summary=dict(kinematic_summary),
        duration_seconds=duration_seconds,
    )
    if b == "template":
        return (
            render_narration_text(
                gesture_label, cast(Dict[str, float | str], dict(kinematic_summary))
            ),
            None,
        )
    if b == "ollama":
        raw = complete_narration_ollama(
            user_prompt=user_prompt,
            base_url=ollama_base_url,
            model=ollama_model,
            timeout_sec=timeout_sec,
        )
        return _sanitize_narration_line(raw, max_words), user_prompt
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
        return _sanitize_narration_line(raw, max_words), user_prompt
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
    """Return a copy of segments with ``narration_text`` replaced when backend != template.

    For non-template backends each segment also gets ``narration_text_template``
    (pre-LLM line) and ``llm_user_prompt`` (full user message sent to the API).
    """
    if (backend or "template").lower() == "template":
        return [dict(s) for s in segments]
    out: list[Dict[str, object]] = []
    for seg in segments:
        start = float(seg["start_time"])
        end = float(seg["end_time"])
        duration = max(1e-3, end - start)
        summary = seg.get("summary")
        if not isinstance(summary, dict):
            raise TypeError("segment missing kinematic summary dict")
        text, user_prompt = synthesize_narration_line(
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
        new["narration_text_template"] = str(seg.get("narration_text", "")).strip()
        new["narration_text"] = text
        new["llm_user_prompt"] = user_prompt
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
