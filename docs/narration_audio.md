# Narration audio (gesture-aligned commentary)

This document describes how **optional narration** works when you enable it in **Streamlit** (`scripts/streamlit_compare.py`) or on the command line via **`scripts/generate_eval_video.py`**.

For a reader-focused explanation of **voice options** and the difference between **template / Ollama / Hugging Face vs Bark**, see [audio_generation.md](audio_generation.md).

Narration is **not** part of the Stable Diffusion sampling loop. It runs **after** the eval video frames are written: the pipeline builds **time segments** (one per contiguous gesture), produces **one line of spoken text per segment**, synthesizes **speech** with **Bark**, aligns audio to segment times, then **muxes** the track onto the MP4(s) with **ffmpeg**.

---

## Two separate models (do not confuse them)

| Stage | What it does | Typical “model” |
|--------|----------------|------------------|
| **Narration text** | One short clinical line per gesture segment, from kinematic summaries (and optional LLM). | **Template** (no LLM), **Ollama** (local chat), or **Hugging Face** (cloud chat via router API). |
| **Text-to-speech** | Turns `narration_text` into waveform audio. | **Bark** (`suno/bark`) — the only supported TTS backend in this repo. |

The **Hugging Face model id** you enter in the UI (or `--hf_narration_model` on the CLI) is used only for **chat completions** that return **plain text**. It does **not** replace Bark for voice synthesis.

---

## Where the code lives

| Piece | Location |
|--------|-----------|
| Segment collapse, kinematic summary, template strings / LLM prompt | `src/suturing_pipeline/audio/narration_templates.py` |
| Ollama + Hugging Face text backends | `src/suturing_pipeline/audio/llm_narration.py` |
| Bark, duration fit, optional OR ambience, WAV track build | `src/suturing_pipeline/audio/tts_converter.py` (re-exported from `tts.py`) |
| Orchestration after frame export | `scripts/generate_eval_video.py` |
| UI checkboxes and subprocess flags | `scripts/streamlit_compare.py` |

---

## End-to-end flow (`generate_eval_video.py`)

1. **Video** — `real.mp4`, `generated.mp4`, and optionally `sidebyside.mp4` are produced as usual.

2. **Segments** — Per-frame metadata is collapsed into **contiguous gesture segments** with `start_time` / `end_time` aligned to the output timeline.

3. **Payload per segment** — For each segment, `build_narration_payload` computes a **kinematic summary** and default **`narration_text`** (template style).

4. **Optional LLM rewrite** — If `--narration_backend` is not `template`, `apply_llm_narration_to_segments` replaces `narration_text` for every segment:
   - **ollama**: HTTP to `ollama_base_url` `/api/chat`.
   - **huggingface** / **hf**: HTTP `POST` to `https://router.huggingface.co/v1/chat/completions` with `Authorization: Bearer <token>`. Requires **`HF_TOKEN`** (environment or `--hf_token`). The response is sanitized (first line, word cap) so TTS stays short.

5. **Artifacts** — Under the run output directory:
   - `narration_segments.json` — full segment dicts: final `narration_text`, `summary`, `gesture` / `gesture_description`, `source_frames`, per-frame **`kinematics_values`** (the same rows used to build the summary), and when `--narration_backend` is not `template`: **`llm_user_prompt`** (exact user message to the chat API) and **`narration_text_template`** (pre-LLM template line).
   - `narration_transcript.txt` — human-readable timed lines (`[start – end] gesture: text`) using the final spoken text.

6. **Bark track** — `synthesize_narration_audio` runs Bark per segment, **time-stretches / trims** each clip to the segment duration, places clips on a single timeline (same sample rate as Bark), optionally mixes **OR ambience** if enabled, then writes the narration audio file used for muxing.

7. **Mux** — `mux_audio_to_video` uses **ffmpeg** to copy the video stream, encode **AAC** from the narration track, and attach it to `generated.mp4` and/or `sidebyside.mp4` depending on flags (either **overwriting** the default files or writing **`*_narrated.mp4`** side files).

---

## Streamlit vs CLI

**Streamlit** toggles mirror the CLI: enabling narration adds `--enable_narration`, `--tts_provider`, `--tts_voice`, optional `--narrate_sidebyside`, `--narration_default_outputs`, and when the text backend is Hugging Face, `--hf_narration_model` plus **`HF_TOKEN` in the child process environment** if you pasted a token (the logged shell command intentionally does not echo the token).

After a run (or when loading a previous run), use **Download narration exports** for `narration_transcript.txt` and `narration_segments.json`. The `--compare` path also writes `<trial_name>_shared_narration_segments.json` and `<trial_name>_shared_narration_transcript.txt` in the same folder; those appear in the same expander when present.

For a full flag list and defaults, run:

```bash
python scripts/generate_eval_video.py --help
```

---

## Requirements and caveats

- **ffmpeg** must be available for muxing (`mux_audio_to_video`).
- **Bark** runs on the **same machine** that runs Python (CUDA / MPS / CPU per your PyTorch install). Large GPU memory use is possible.
- **Hugging Face narration** needs a valid **read token** and a **model that is actually served** on the router/chat path; gated models require accepting the license on the Hub for the account that owns the token. Errors from the API are surfaced in the generator log / metadata.
- **Ollama** on `127.0.0.1` refers to the machine running the script (not your laptop if the code runs on a remote VM or Colab). Use Hugging Face there, run Ollama on that host, or point `--ollama_base_url` at a reachable server.

---

## Related flags (quick reference)

- `--enable_narration` — turn the narration pipeline on after video render.
- `--narration_backend` — `template` | `ollama` | `huggingface`.
- `--hf_narration_model` — Hub model id for **text** only (chat).
- `--tts_provider` / `--tts_voice` — Bark preset (e.g. `v2/en_speaker_9`).
- `--narration_default_outputs` — mux into the canonical `generated.mp4` / `sidebyside.mp4` vs separate `*_narrated.mp4` files.
- `--compare` — separate path that builds a **shared** narration bundle for dual-video comparison (see `generate_eval_video.py` and `compositor.py`).

For behaviour of individual functions, see docstrings in the modules listed above.

---

## Speech evaluation (WER, accuracy, summary quality)

To define human rubrics, automatic WER protocol, and the `eval.speech` fields read by Streamlit, see **[speech_eval_rubric.md](speech_eval_rubric.md)**.
