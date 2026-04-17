---
name: JIGSAWS SD Training Setup
overview: "Build five new modules for kinematic-conditioned Stable Diffusion training: JIGSAWS data parsers, a PyTorch dataset, a kinematic encoder, an SD training script, and an inference script. All integrate into the existing `src/suturing_pipeline/` package."
todos:
  - id: data-utils
    content: Create src/suturing_pipeline/data/data_utils.py with 6 JIGSAWS parsing functions
    status: completed
  - id: dataset
    content: Create src/suturing_pipeline/data/jigsaws_dataset.py with JIGSAWSDataset class
    status: completed
  - id: encoder
    content: Create src/suturing_pipeline/synthesis/kinematic_encoder.py with KinematicEncoder module
    status: completed
  - id: train-script
    content: Create scripts/train_sd.py with full SD training loop (LoRA + full modes)
    status: completed
  - id: inference-script
    content: Create scripts/inference_sd.py with DDIM sampling pipeline
    status: completed
  - id: deps
    content: Update requirements.txt with diffusers, transformers, accelerate, peft, safetensors, scikit-learn
    status: completed
isProject: false
---

# JIGSAWS Kinematic-Conditioned SD — Implementation Plan

## File Layout

New files slotted into the existing package structure:

```
src/suturing_pipeline/
  data/
    data_utils.py           <-- Step 1: JIGSAWS parsers
    jigsaws_dataset.py      <-- Step 2: PyTorch Dataset
  synthesis/
    kinematic_encoder.py    <-- Step 3: KinematicEncoder nn.Module
scripts/
  train_sd.py               <-- Step 4: SD training loop
  inference_sd.py           <-- Step 5: DDIM inference
requirements.txt            <-- updated with new deps
```

---

## Step 1 — `data_utils.py`

**File:** [src/suturing_pipeline/data/data_utils.py](src/suturing_pipeline/data/data_utils.py) (new)

Six pure-function utilities for JIGSAWS file formats:

- `parse_metafile(path)` — parse `metafile.txt` (whitespace-delimited, columns: trial_name, score, skill_level). Return `{trial_name: skill_level}`.
- `parse_transcription(path)` — parse gesture transcription files (`START END GESTURE` per line). Return list of `(start, end, label)` tuples.
- `parse_kinematics(path)` — `np.loadtxt(path)` returning `(N, 76)` array. JIGSAWS kinematics files are headerless, whitespace-delimited, 76 columns.
- `load_split(path)` — read `train.txt` / `test.txt`, one trial name per line. Return list of strings.
- `get_frame_label_map(transcription)` — expand `(start, end, label)` tuples into `{frame_idx: gesture_label}` for every frame in every range (inclusive).
- `filter_expert_trials(metafile_path)` — calls `parse_metafile`, returns trial names where `skill_level == "E"` (JIGSAWS uses single-letter codes: `N`/`I`/`E` in the third column, or sometimes the full word — we handle both).

**Note:** The existing [src/suturing_pipeline/data/alignment.py](src/suturing_pipeline/data/alignment.py) has `read_kinematics_file` using pandas with header sniffing. JIGSAWS raw kinematics are headerless 76-column files, so `parse_kinematics` will use `np.loadtxt` directly — simpler and more correct for this format.

---

## Step 2 — `jigsaws_dataset.py`

**File:** [src/suturing_pipeline/data/jigsaws_dataset.py](src/suturing_pipeline/data/jigsaws_dataset.py) (new)

`JIGSAWSDataset(torch.utils.data.Dataset)` with:

- **Init args:** `data_root, task, split, split_type="onetrialout", balance="balanced", itr=1, expert_only=False, modality="both", image_size=256`
- **Path resolution** (derived from JIGSAWS conventions):
  - Split file: `{data_root}/experimental_setup/{task}/{balance}/gestureclassification/{split_type}/one_out/itr_{itr}/{split}.txt`
  - Video: `{data_root}/{task}/video/{TaskName}_{TrialSuffix}_capture1.avi` (try both captures)
  - Kinematics: `{data_root}/{task}/kinematics/allgestures/{TaskName}_{TrialSuffix}.txt`
  - Transcription: `{data_root}/{task}/transcriptions/{TaskName}_{TrialSuffix}.txt`
  - Metafile: `{data_root}/{task}/metafile.txt`
- **Init logic:**
  1. Load trial list via `load_split`; optionally filter to experts via `filter_expert_trials`
  2. For each trial: parse kinematics (store numpy array), build frame-label map, open video to get frame count
  3. Build a flat index: list of `(trial_idx, frame_idx)` tuples covering every frame across all trials
  4. Fit a `sklearn.preprocessing.StandardScaler` on all kinematics rows (concatenated) for normalization
  5. Build gesture-label-to-int mapping from all unique gesture labels seen (sorted: G1->0, G2->1, ...)
- `**__getitem`:
  1. Look up `(trial_idx, frame_idx)` from flat index
  2. Seek video to frame with `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, frame_idx)` + `read()` — lazy, no preload
  3. Resize to `image_size`, convert BGR->RGB, normalize to [-1, 1] as float32 tensor `[3, H, W]`
  4. Transform kinematics row via fitted scaler -> `torch.float32` tensor `[76]`
  5. Map gesture label to int
  6. Return `(frame, kinematics, gesture_label)`
- `**__len`: total frames across all trials

**Design note:** VideoCapture seeking per `__getitem` call is not the fastest but satisfies the "lazy, no preload" requirement. For real training speed, a future optimization would pre-extract frames to disk. The scaler is fit at init on all kinematics data for the split, which is fine since JIGSAWS kinematics files are small (a few MB total).

---

## Step 3 — `kinematic_encoder.py`

**File:** [src/suturing_pipeline/synthesis/kinematic_encoder.py](src/suturing_pipeline/synthesis/kinematic_encoder.py) (new)

`KinematicEncoder(nn.Module)`:

- **Init args:** `kin_dim=76, num_gestures=16, seq_len=77, embed_dim=768`
- **Components:**
  - `mlp`: `nn.Sequential(Linear(76, 256), GELU, Linear(256, 512), GELU, Linear(512, seq_len * embed_dim))`
  - `gesture_embed`: `nn.Embedding(num_gestures, embed_dim)`
  - `layer_norm`: `nn.LayerNorm(embed_dim)` applied after reshape
- **Forward(kinematics, gesture_label):**
  1. `mlp(kinematics)` -> reshape to `[B, 77, embed_dim]`
  2. `gesture_embed(gesture_label)` -> `[B, embed_dim]`
  3. Add gesture embedding to token 0: `out[:, 0, :] += gesture_emb`
  4. `layer_norm(out)` -> return `[B, 77, embed_dim]`

The output shape `[B, 77, 768]` matches the CLIP text encoder output shape expected by SD 1.x's U-Net cross-attention. For SD 2.x, set `embed_dim=1024`.

---

## Step 4 — `train_sd.py`

**File:** [scripts/train_sd.py](scripts/train_sd.py) (new)

CLI training script using `argparse` with flags: `--data_root, --task, --split_type, --itr, --batch_size, --lr, --epochs, --model_id` (default `"runwayml/stable-diffusion-v1-5"`), `--train_mode` (`lora` | `full`), `--save_dir, --save_every, --expert_only, --image_size, --gradient_accumulation_steps`

**Training logic:**

1. **Load SD components** from `model_id` via diffusers:

- `AutoencoderKL` (VAE) — frozen
- `UNet2DConditionModel` — frozen base; if `--train_mode lora`, apply LoRA to cross-attention layers via `peft`; if `full`, unfreeze all cross-attention `to_k` / `to_v` projection layers
- `DDPMScheduler` for training noise schedule

1. **Instantiate** `KinematicEncoder` with `embed_dim` matching the model (768 for SD 1.x, 1024 for SD 2.x — auto-detected from `unet.config.cross_attention_dim`)
2. **Build dataset** via `JIGSAWSDataset(data_root, task, split="train", expert_only=args.expert_only, image_size=args.image_size)`
3. **Training loop** (epochs x batches):

- Encode frame -> latent `z` via frozen VAE encoder, scale by `vae.config.scaling_factor`
- Sample random timestep `t`, add noise to `z` via scheduler
- Encode kinematics + gesture -> `encoder_hidden_states` via `KinematicEncoder`
- Predict noise via U-Net: `unet(noisy_z, t, encoder_hidden_states).sample`
- MSE loss between predicted and actual noise
- Backprop through `KinematicEncoder` + unfrozen U-Net parameters only

1. **Logging:** tqdm progress bar with running loss
2. **Checkpointing:** every `--save_every` steps, save `KinematicEncoder` state dict + LoRA weights (or cross-attn weights) + optimizer state + scaler params (from dataset) + gesture label mapping

---

## Step 5 — `inference_sd.py`

**File:** [scripts/inference_sd.py](scripts/inference_sd.py) (new)

CLI args: `--checkpoint, --model_id, --kinematics_file, --gesture_label, --output_path, --num_inference_steps` (default 50), `--guidance_scale, --image_size`

**Inference logic:**

1. Load SD pipeline components (VAE, U-Net, scheduler — use `DDIMScheduler` for fast sampling)
2. Load `KinematicEncoder` + saved LoRA/cross-attn weights from checkpoint
3. Load + normalize kinematics via saved scaler params from checkpoint
4. Encode kinematics + gesture label -> conditioning embeddings
5. DDIM denoising loop: start from random noise, iteratively denoise conditioned on kinematic embeddings
6. Decode final latent via VAE decoder -> image
7. Save as `.png`

---

## Dependency Updates

Add to [requirements.txt](requirements.txt):

```
diffusers
transformers
accelerate
peft
safetensors
scikit-learn
```

---

## Key Design Decisions

- **Separate from existing modules:** The new `data_utils.py` and `jigsaws_dataset.py` are purpose-built for JIGSAWS raw format and SD training. The existing `loader.py` / `alignment.py` serve the detection pipeline and remain untouched.
- **No text prompts:** The `KinematicEncoder` fully replaces the CLIP text encoder. The SD text encoder is never loaded or used.
- **Scaler serialization:** The `StandardScaler` fitted during dataset init is saved with the checkpoint so inference can normalize new kinematics identically.
- **Gesture label mapping:** Saved with checkpoint to ensure consistent `G1->0, G2->1, ...` mapping at inference.
- **LoRA vs full:** LoRA (via `peft`) is the recommended default — trains ~1% of U-Net params, much faster. Full cross-attention fine-tuning is available as a fallback.

---

## Sample Commands

**Training:**

```bash
python scripts/train_sd.py \
  --data_root /path/to/JIGSAWS \
  --task knot_tying \
  --expert_only \
  --train_mode lora \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 50 \
  --save_dir ./checkpoints/knot_tying_expert_lora
```

**Inference:**

```bash
python scripts/inference_sd.py \
  --checkpoint ./checkpoints/knot_tying_expert_lora/step_5000.pt \
  --kinematics_file /path/to/JIGSAWS/knot_tying/kinematics/allgestures/Knot_Tying_B001.txt \
  --gesture_label 3 \
  --output_path ./generated_frame.png \
  --num_inference_steps 50
```
