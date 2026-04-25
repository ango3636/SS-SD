# Audio generation: how it works, voices, and “which model?”

This README is about **spoken narration** on eval videos (Streamlit or `scripts/generate_eval_video.py`). It clears up a common confusion: **template**, **Ollama**, and **Hugging Face** are **not** alternatives to **Bark**. They control **different steps**.

---

## Two separate choices (stacked, not pick-one)

| Layer | What you choose | Role |
|--------|------------------|------|
| **1. Narration text** | `template` · `ollama` · `huggingface` | How each gesture segment gets its **one line of words** to speak (`narration_text`). |
| **2. Speech (TTS)** | **`bark` only** in this repo | Turns that text into **waveform audio** (Suno **Bark** via Hugging Face `suno/bark`). |

So: **Bark is always the voice** when narration is enabled. **Template / Ollama / Hugging Face** only decide **who writes the script** before Bark reads it.

---

## What people mean by “default”

The word **default** shows up in two places; they are unrelated.

| “Default” | Meaning here |
|-----------|----------------|
| **Default narration text** | Backend **`template`** — deterministic sentences built from kinematic summaries in code (`narration_templates.py`). No LLM, no API key. This is the **default** `--narration_backend` in the CLI and the first option in Streamlit (“Narration text (before Bark)”). |
| **Default voice** | Bark preset **`v2/en_speaker_9`** — `--tts_voice` default and the Streamlit placeholder. Calm, English, “clinical instructor” style per in-app help. |

There is **no** separate audio engine called “default”; silent videos are simply **narration off**. Turning narration on always uses **Bark** for sound.

---

## How audio is actually generated (short version)

1. After **real / generated / side-by-side** video frames exist, the run **collapses** frames into **gesture segments** with start/end times on the output timeline.
2. For each segment, the pipeline builds a **kinematic summary** and an initial line of text.
3. If the text backend is **`template`**, that line stays the **template** wording. If it is **`ollama`** or **`huggingface`**, an **LLM** rewrites the line (same clinical style constraints, word cap), using only **text** APIs — not Bark.
4. **Bark** synthesizes **one clip per segment** from the final `narration_text`, **time-stretches / trims** clips so they fit segment length, and lays them on a **single timeline** (24 kHz). Optional **`--or_ambience`** can mix generated OR background under the voice (separate AudioGen model; see `generate_eval_video.py --help`).
5. **ffmpeg** muxes that track into the chosen MP4(s) as AAC.

For file names, flags, and module map, see [narration_audio.md](narration_audio.md).

---

## Voice options (Bark only)

**Narration provider** in Streamlit is fixed to **`bark`** — there is no second TTS vendor in the UI.

What you can change is the **Bark voice preset** (`--tts_voice` / “Bark voice preset” in Streamlit):

- **Preset string** — Passed straight to Bark’s processor as `voice_preset` (see `BarkTTSConverter` in `src/suturing_pipeline/audio/tts_converter.py`). The hub model is **`suno/bark`**; presets follow Bark’s usual naming, e.g. English **`v2/en_speaker_0`** … **`v2/en_speaker_9`**.
- **Built-in default** — **`v2/en_speaker_9`** (`DEFAULT_VOICE_PRESET`).
- **Alternate suggested in CLI help** — **`v2/en_speaker_6`** as another English speaker.
- **Other languages / speakers** — Any preset string **supported by your installed Bark/processor** for that checkpoint (e.g. other `v2/...` locales if available). If a string is invalid for the model, Bark/transformers will error at runtime.

**`--voice_preset`** on `generate_eval_video.py` is an **alias** for `--tts_voice` and is also used by **`--compare`** for the shared Bark track.

---

## Template vs Ollama vs Hugging Face (text only)

These appear as **“Narration text (before Bark)”** in Streamlit (`--narration_backend` on the CLI).

| Backend | What it does | Typical requirements | Tradeoffs |
|---------|----------------|----------------------|-----------|
| **`template`** | Fills `narration_text` from **fixed rules + kinematics** (same gesture + similar motion → similar sentence). | JIGSAWS kinematics available for the segment. | **Reproducible**, offline, fast, no keys. Less variety than an LLM. |
| **`ollama`** | Sends a **prompt** (gesture, description, kinematic bullets, word budget) to a **local** Ollama **`/api/chat`** endpoint; response becomes `narration_text` after sanitization. | Ollama running where **Python** runs (`ollama serve`, model pulled, e.g. `llama3.2`). **`--ollama_base_url`** / **`--ollama_model`**. | **Free**, private, no Hugging Face bill; **127.0.0.1** is the **machine running the script** (not your laptop if the job is on a VM/Colab). |
| **`huggingface`** / **`hf`** | Same style of prompt, but completion via **Hugging Face router**: `https://router.huggingface.co/v1/chat/completions` with **`HF_TOKEN`** (or **`--hf_token`**). Model id from **`--hf_narration_model`** (default in code: `meta-llama/Llama-3.2-1B-Instruct`). | Valid token; model **routed** for chat; **gated** models need license accepted on the Hub for that account. | Good for **cloud** or when Ollama is not on the runner; subject to **API limits** and **network** policy. |

**Important:** the Hugging Face **chat model id** is **not** your TTS voice. It only generates **text**. **Bark** still produces all speech.

---

## Quick reference diagram

```
[Gesture segment + kinematics]
           │
           ▼
   ┌───────────────────┐
   │ Narration text    │  template  →  rules in code
   │ (one line/segment)│  ollama    →  local HTTP chat
   │                   │  huggingface → HF router chat
   └─────────┬─────────┘
             │  narration_text
             ▼
   ┌───────────────────┐
   │ Bark (suno/bark)  │  voice = --tts_voice (e.g. v2/en_speaker_9)
   │ TTS + time fit    │
   └─────────┬─────────┘
             │  WAV / track
             ▼
        ffmpeg → AAC muxed into MP4
```

---

## CLI flags tied to this README

| Flag | Purpose |
|------|---------|
| `--enable_narration` | Run the narration pipeline after video render. |
| `--narration_backend` | `template` \| `ollama` \| `huggingface` \| `hf` — **text** source. |
| `--tts_provider` | Must be **`bark`** (only supported value). |
| `--tts_voice` / `--voice_preset` | **Bark voice preset** string. |
| `--ollama_base_url`, `--ollama_model` | Local LLM when backend is `ollama`. |
| `--hf_narration_model`, `--hf_token` / `HF_TOKEN` | Cloud LLM when backend is `huggingface`. |
| `--or_ambience` | Optional OR background bed under Bark (AudioGen; extra deps). |

Run `python scripts/generate_eval_video.py --help` for defaults and edge-case flags.
