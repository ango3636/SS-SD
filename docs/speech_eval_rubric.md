# Speech evaluation rubric (narration / TTS)

This document defines **how to score** the optional speech metrics shown in Streamlit under **Evaluation metrics → Narration generation → Speech evaluation**, and how to record them in each run’s **`metadata.json`**.

The UI reads optional nested keys (see [JSON fields](#json-fields-metadatajson)):

```json
"eval": {
  "speech": {
    "wer": null,
    "accuracy": null,
    "summary_quality": null
  }
}
```

`generate_eval_video.py` fills **`eval.speech.wer`** automatically when narration succeeds: it runs **Whisper-tiny** ASR on `narration_track.m4a` and compares to the intended script (same policy as “Intended script” below). Use `--skip_speech_eval` to skip. **Accuracy** and **summary quality** remain for **human** scoring unless you extend the pipeline.

---

## Roles of the three metrics

| Field | Typical source | What it measures |
|--------|----------------|------------------|
| **`wer`** | Automatic (ASR + reference text) | How much the **spoken** output diverges from a **reference transcript** (lower is better). |
| **`accuracy`** | Human checklist or rule-based tally | Whether narration **content** matches what happened (gestures, steps, safe clinical wording). |
| **`summary_quality`** | Human Likert (this rubric) | How **good** the commentary is as teaching / summary audio, independent of exact word match. |

Use **WER** when you care about fidelity to a fixed script. Use **accuracy** when exact words can differ but **facts** must be right. Use **summary_quality** for **holistic** usefulness and polish.

---

## 1. Word error rate (`eval.speech.wer`)

WER is **not** a subjective rubric. It is a standard string edit metric between **reference** and **hypothesis** text after the same normalization (e.g. lowercasing, collapsing whitespace, optional removal of punctuation).

### Reference text (pick one policy and stick to it)

- **Intended script** — Concatenate the per-segment lines your pipeline **meant** to speak (from `narration_segments.json` / `narration_transcript.txt` in the run directory), **or**
- **Gold transcript** — JIGSAWS (or other) **human transcription** for the same time span, if aligned to the eval clip.

Document which policy you used in your study notes or in a sibling field (e.g. `eval.speech.wer_reference` as a short string) if you extend metadata beyond what the UI shows.

### Hypothesis text

- Run **automatic speech recognition** on the **narration** waveform (or on the mixed video track if you accept more noise).
- Use the same language model / decoding settings for all runs in a comparison.

### Computation

- Normalize reference and hypothesis, then compute WER (substitutions + insertions + deletions, divided by reference length).
- Store a **scalar** in `eval.speech.wer`, conventionally in **[0, 1]** (e.g. `0.12` for 12% WER). If you prefer **percent**, document that; the Streamlit formatter will still display the number.

### Interpretation (informative, not prescriptive)

| WER (order of magnitude) | Rough read |
|--------------------------|------------|
| &lt; 0.10 | Strong match to reference wording. |
| 0.10–0.25 | Usable; noticeable word changes. |
| &gt; 0.25 | Large divergence; check ASR errors vs true mis-speaks. |

---

## 2. Content accuracy (`eval.speech.accuracy`)

**Accuracy** here means **correctness of informational content** relative to a gold standard you define—not WER.

### Suggested operational definition

For each **narration segment** (or each **sentence** if you split further):

1. **Correct** — Describes the right gesture / phase, no contradictions with the video or transcript, no unsafe false clinical claims.
2. **Incorrect** — Wrong gesture, wrong instrument action, omission of a critical step you required, or a clear factual error.

**Score:** `accuracy = (# correct segments) / (# segments scored)` in **[0, 1]**, or store **percent correct** in **[0, 100]** and note that in your paper; be consistent across runs.

### Optional stricter rubric (per segment)

| Code | Meaning |
|------|---------|
| C | Correct: matches gold **and** is clinically plausible. |
| P | Partial: mostly right but minor omission or imprecise wording. |
| W | Wrong: clear mismatch with gold or video. |

You may map `P` to 0.5 in a partial-credit scheme, or collapse `P` into correct/incorrect—**define before rating**.

### Raters

- At least **one** expert rater familiar with suturing nomenclature; **two** independent raters are better for publications.
- Resolve disagreements with a short adjudication rule (e.g. third rater, or majority).

---

## 3. Summary quality (`eval.speech.summary_quality`)

Use a **single overall** 1–5 score (or mean of dimensions—if you use dimensions, **predefine** whether the UI stores the mean only or you keep per-dimension scores outside `metadata.json`).

### Dimensions (rate each 1–5, then mean → `summary_quality`)

**1 = poor · 2 = below average · 3 = acceptable · 4 = good · 5 = excellent**

| Dimension | Question for the rater | Anchor notes |
|-----------|-------------------------|--------------|
| **Fluency** | Does it sound like natural speech (grammar, pacing when listened to)? | 1 = broken / unreadable; 5 = smooth, broadcast-like. |
| **Coherence** | Do ideas follow in a sensible order for the clip? | 1 = disjointed; 5 = clear thread through the segment. |
| **Coverage** | Does it mention the main actions visible in that span? | 1 = misses the point; 5 = captures key moves without fluff. |
| **Clinical plausibility** | Could a trainee trust the wording in a teaching context? | 1 = misleading or unsafe tone; 5 = appropriately cautious and accurate. |
| **Alignment to video** | Does what is said match what is seen at that time? | 1 = frequent mismatch; 5 = tight alignment. |

**`summary_quality`** = mean of the five dimension scores (or your chosen subset—document if you drop a dimension).

### Single-rater shortcut (pilots only)

One rater gives one overall 1–5 **holistic** score using the dimensions above as a mental checklist. For anything publishable, prefer **two raters** and report mean of the two overall scores (or ICC).

---

## JSON fields (`metadata.json`)

Place alongside other top-level keys in the run’s `metadata.json`:

```json
{
  "eval": {
    "speech": {
      "wer": 0.14,
      "accuracy": 0.86,
      "summary_quality": 4.2
    }
  }
}
```

Optional keys you may add for traceability (ignored by the current Streamlit board unless you extend the UI):

```json
"eval": {
  "speech": {
    "wer": 0.14,
    "wer_reference": "intended_script",
    "wer_asr_model": "whisper-large-v3",
    "accuracy": 0.86,
    "accuracy_definition": "segment_correct_fraction",
    "summary_quality": 4.2,
    "summary_quality_scale": "1-5_mean_of_5_dims",
    "rater_ids": ["r1", "r2"],
    "adjudication": "majority"
  }
}
```

---

## Workflow checklist

1. Freeze **reference policy** for WER and **gold** for accuracy.
2. Run ASR → compute WER → write `eval.speech.wer`.
3. Rate segments for accuracy → write `eval.speech.accuracy`.
4. Rate summary quality (dimensions → mean) → write `eval.speech.summary_quality`.
5. Reload the run in Streamlit (or re-open **Previous runs**) so the app reads the updated `metadata.json`.

---

## Related

- Narration pipeline and artefacts: [narration_audio.md](narration_audio.md)
- Music / foley-style metrics use **`eval.sound`** in the same `metadata.json` (see Streamlit **Sound effects** tab captions in `scripts/streamlit_compare.py`).
