"""Render a real-vs-generated video comparison for a trained SD checkpoint.

Picks one held-out JIGSAWS trial, walks a contiguous range of frames in
temporal order, and for each frame:

* reads the real RGB frame from the ``captureN.avi`` file,
* feeds the aligned 76-dim kinematics row + gesture label into
  :class:`~suturing_pipeline.synthesis.sd_sampler.SDSampler`,
* decodes the generated frame to RGB.

Three MP4s are written:

* ``real.mp4`` - real frames only.
* ``generated.mp4`` - generated frames only.
* ``sidebyside.mp4`` - horizontally stacked (REAL | GEN).

A ``metadata.json`` records run args, the trial chosen, per-frame
gesture labels, and wall-clock timing.

Notes
-----
The SD model was trained with ``frame_stride=90`` on a single frame per
step; it has no explicit temporal-consistency term, so expect flicker
between adjacent generated frames.  Using ``--fixed_seed`` (default)
re-uses the same noise latent for every frame, which empirically gives
the most coherent output under these constraints.  For long clips,
chaining ``--anchor_mode prev_gen`` or ``flow_warp`` can still drift or
smear; use ``--anchor_reset_every N`` to periodically drop the anchor
and resample from noise (reduces compound error, at the cost of rarer
sharp transitions).

Example
-------
Smoke test (~60 generated frames at 25 steps each)::

    python scripts/generate_eval_video.py \\
        --checkpoint checkpoints/suturing_expert_lora/step_1480.pt \\
        --data_root  /Users/amyngo/SS-SD/data/gdrive_cache \\
        --num_frames 60 --frame_step 6 --num_inference_steps 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.audio.compositor import (
    generate_shared_audio_track,
    mux_audio_to_generated_video,
    mux_audio_to_raw_video,
    side_by_side_comparison,
)
from suturing_pipeline.audio.llm_narration import apply_llm_narration_to_segments
from suturing_pipeline.audio.narration_templates import (
    build_expert_speed_stats,
    build_narration_payload,
    collapse_frame_records,
    kinematics_segment_to_jsonable,
    write_narration_transcript,
)
from suturing_pipeline.audio.tts import (
    DEFAULT_VOICE_PRESET,
    BarkTTSConverter,
    mux_audio_to_video,
    synthesize_narration_audio,
)
from suturing_pipeline.data.jigsaws_dataset import JIGSAWSDataset
from suturing_pipeline.synthesis.sd_sampler import SDSampler


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a real-vs-generated video comparison."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a train_sd.py checkpoint (.pt file).",
    )
    p.add_argument(
        "--data_root",
        required=True,
        help="JIGSAWS root (same one used for training).",
    )
    p.add_argument("--output_dir", default="./outputs/eval_video")
    p.add_argument(
        "--run_name",
        default=None,
        help="Sub-folder under --output_dir. Defaults to a timestamp.",
    )

    # Split selection - defaults read from checkpoint if omitted.
    p.add_argument("--split_type", default=None)
    p.add_argument("--balance", default=None)
    p.add_argument("--held_out", type=int, default=None)
    p.add_argument("--itr", type=int, default=None)
    p.add_argument("--capture", type=int, default=None, choices=[1, 2])
    p.add_argument(
        "--expert_only",
        action="store_true",
        help=(
            "Restrict to expert trials: meta file column 2 (1-based), "
            "self-reported E — not the GRS column. See jigsaws_metafile_layout."
        ),
    )
    p.add_argument(
        "--dataset_split",
        choices=["train", "test"],
        default="test",
        help="Which JIGSAWS split file to use (Train.txt vs Test.txt). "
        "Expert-only *test* splits may be empty; use --dataset_split train "
        "with --expert_only for many expert trials.",
    )

    # Trial / frame-range selection
    p.add_argument(
        "--trial_name",
        default=None,
        help="Specific trial to render (e.g. Suturing_B001). Defaults "
        "to the first trial in the resolved split. Incompatible with "
        "--random_trial.",
    )
    p.add_argument(
        "--random_trial",
        action="store_true",
        help="Pick one trial uniformly at random from the resolved trial list.",
    )
    p.add_argument(
        "--random_trial_seed",
        type=int,
        default=None,
        help="RNG seed for --random_trial (omit for a nondeterministic pick).",
    )
    p.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Starting frame index in the chosen trial's video.",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=120,
        help="How many OUTPUT frames to render (after applying --frame_step). "
        "Ignored when --duration_seconds is set.",
    )
    p.add_argument(
        "--duration_seconds",
        type=float,
        default=None,
        help="If set, sets the output frame count to "
        "ceil(duration_seconds * fps_out) so the MP4 is about this long.",
    )
    p.add_argument(
        "--frame_step",
        type=int,
        default=6,
        help="Take every Nth source frame (default 6, ~5 fps from 30 fps).",
    )

    # Sampling / output
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--fps_out", type=float, default=5.0)
    p.add_argument(
        "--seed", type=int, default=0, help="Base seed for noise latents."
    )
    p.add_argument(
        "--vary_seed",
        action="store_true",
        help="Use seed+frame_idx per frame instead of a fixed seed. "
        "Increases diversity but also flicker.",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no_sidebyside",
        action="store_true",
        help="Skip the stacked sidebyside.mp4 artefact.",
    )
    p.add_argument(
        "--enable_narration",
        action="store_true",
        help="Generate a narration track and mux narrated MP4 artefacts.",
    )
    p.add_argument(
        "--tts_provider",
        default="bark",
        help="TTS backend (currently: bark).",
    )
    p.add_argument(
        "--tts_voice",
        default=DEFAULT_VOICE_PRESET,
        help=(
            "Bark voice preset, e.g. 'v2/en_speaker_9' (default) or "
            "'v2/en_speaker_6' (alternate clinical tone)."
        ),
    )
    p.add_argument(
        "--voice_preset",
        default=None,
        help=(
            "Alias for --tts_voice, also used by --compare for the shared "
            "Bark narration track."
        ),
    )
    p.add_argument(
        "--narration_min_segment_sec",
        type=float,
        default=0.5,
        help="Minimum narrated segment duration in seconds.",
    )
    p.add_argument(
        "--narration_min_empirical_count",
        type=int,
        default=8,
        help="Min expert segment count before using gesture-specific empirical speed thresholds.",
    )
    p.add_argument(
        "--disable_empirical_speed_stats",
        action="store_true",
        help="Force absolute speed thresholds instead of expert-derived gesture stats.",
    )
    p.add_argument(
        "--narrate_sidebyside",
        action="store_true",
        help="Also mux narration onto sidebyside.mp4 when it is produced.",
    )
    p.add_argument(
        "--narration_default_outputs",
        action="store_true",
        help=(
            "Mux narration directly into generated.mp4 (and sidebyside.mp4 "
            "when --narrate_sidebyside is set) instead of creating "
            "*_narrated.mp4 copies."
        ),
    )
    p.add_argument(
        "--allow_narration_failure",
        action="store_true",
        help=(
            "When narration is enabled, continue even if TTS/mux fails. "
            "By default, narration failures stop the run so silent outputs "
            "do not look like success."
        ),
    )
    p.add_argument(
        "--or_ambience",
        action="store_true",
        help=(
            "Mix facebook/audiogen-medium OR ambience under Bark per gesture "
            "segment (-18 dB). Requires audiocraft when narration or --compare "
            "audio is produced."
        ),
    )
    p.add_argument(
        "--foley_dir",
        default=None,
        help=(
            "Optional directory of mono WAV files named G1.wav … G15.wav "
            "(case-insensitive stem). Each file is mixed per matching "
            "narration segment at --foley_gain_db (default -12 dB), aligned "
            "with --foley_align. Missing gestures are skipped. For two layers "
            "on the same label, add Gn_L.wav and Gn_R.wav. For two gesture "
            "classes in one segment, set foley_gestures or foley_gesture_right "
            "in narration_segments.json before synthesis."
        ),
    )
    p.add_argument(
        "--foley_gain_db",
        type=float,
        default=-12.0,
        help="Linear mix level for WAV foley relative to the segment stem (voice + optional ambience).",
    )
    p.add_argument(
        "--foley_align",
        choices=["start", "center"],
        default="start",
        help="Where to place each foley clip within the stretched segment duration.",
    )
    p.add_argument(
        "--narration_backend",
        choices=["template", "ollama", "huggingface", "hf"],
        default="template",
        help=(
            "How to produce spoken lines before Bark: 'template' (deterministic "
            "kinematic text), 'ollama' (free local LLM, install Ollama), or "
            "'huggingface' / 'hf' (HF Inference API; set HF_TOKEN or --hf_token)."
        ),
    )
    p.add_argument(
        "--ollama_base_url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL when --narration_backend ollama.",
    )
    p.add_argument(
        "--ollama_model",
        default="llama3.2",
        help="Ollama model tag (e.g. llama3.2, mistral, qwen2.5:7b).",
    )
    p.add_argument(
        "--hf_narration_model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help=(
            "HF model id for router /v1/chat/completions. Gated models require accepting "
            "the license on the Hub. See https://huggingface.co/inference/models"
        ),
    )
    p.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face read token (optional if HF_TOKEN is set in the environment).",
    )
    p.add_argument(
        "--narration_llm_timeout_sec",
        type=float,
        default=120.0,
        help="Per-segment HTTP timeout when calling Ollama or Hugging Face.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Produce the dual-audio comparison bundle: a shared Bark "
            "narration WAV muxed independently onto the original "
            "JIGSAWS .avi AND the generated video, plus a labeled "
            "side-by-side comparison MP4. Requires --raw_data_root."
        ),
    )
    p.add_argument(
        "--raw_data_root",
        default=None,
        help=(
            "Path to the JIGSAWS data root used for the raw .avi in "
            "--compare. Defaults to --data_root when omitted."
        ),
    )

    # Temporal anchoring (Tier-1 coherence controls).
    p.add_argument(
        "--anchor_mode",
        default="none",
        choices=["none", "prev_gen", "prev_real", "flow_warp"],
        help=(
            "Initialise each frame's latent from a reference image instead "
            "of pure noise.  'prev_gen' uses the previously GENERATED frame "
            "(classic SDEdit / img2img temporal anchoring).  'prev_real' "
            "uses the previous REAL frame (diagnostic baseline; leaks "
            "ground truth). 'flow_warp' warps the previous generated frame "
            "by dense optical flow between the two real frames, giving the "
            "model motion cues from the real video."
        ),
    )
    p.add_argument(
        "--init_strength",
        type=float,
        default=0.7,
        help=(
            "Fraction of the DDIM schedule to run when anchoring "
            "(1.0 == pure noise, no anchor effect; 0.5-0.75 = strong "
            "temporal coherence; too low freezes motion)."
        ),
    )
    p.add_argument(
        "--anchor_reset_every",
        type=int,
        default=0,
        help=(
            "If > 0, skip temporal anchoring every N output frames (1-based "
            "count within this clip). Reduces long-horizon drift when "
            "chaining prev_gen/flow_warp, at the cost of occasional sharper "
            "transitions. 0 = never reset (default)."
        ),
    )
    return p.parse_args()


def _tensor_to_uint8_rgb(frame_tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``[3, H, W]`` frame tensor in ``[-1, 1]`` to HWC uint8 RGB."""
    img = (frame_tensor.clamp(-1, 1) + 1) / 2 * 255
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _open_video_writer(
    out_path: Path, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    """Open an MP4 writer using the mp4v fourcc (broadly compatible)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {out_path}. Check that your "
            "OpenCV build includes ffmpeg/mp4v support."
        )
    return writer


def _write_rgb_frame(writer: cv2.VideoWriter, rgb: np.ndarray) -> None:
    """cv2 writers expect BGR uint8."""
    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _resolve_trial_index(
    dataset: JIGSAWSDataset, trial_name: Optional[str]
) -> int:
    if trial_name is None:
        return 0
    try:
        return dataset._trial_names.index(trial_name)
    except ValueError as e:
        raise SystemExit(
            f"Trial {trial_name!r} not found in test split. "
            f"Available: {dataset._trial_names}"
        ) from e


def _plan_frame_indices(
    frame_count: int, start: int, step: int, num_out: int
) -> List[int]:
    """List of source frame indices to sample, clipped to video length."""
    step = max(1, step)
    planned = [start + i * step for i in range(num_out)]
    return [f for f in planned if 0 <= f < frame_count]


def _warp_by_flow(
    source_rgb: np.ndarray,
    real_prev_rgb: np.ndarray,
    real_curr_rgb: np.ndarray,
) -> np.ndarray:
    """Warp ``source_rgb`` by the dense optical flow
    ``real_prev_rgb -> real_curr_rgb``.

    Used to turn the previous generated frame into a motion-compensated
    estimate of the current frame, which then serves as the img2img init.
    """
    prev_gray = cv2.cvtColor(real_prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(real_curr_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    # cv2.remap expects the source coordinate each output pixel pulls from,
    # so subtract the flow rather than add it.
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    return cv2.remap(
        source_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def main() -> None:
    args = _parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved: Dict = ckpt.get("args", {}) or {}

    split_type = args.split_type or saved.get("split_type", "onetrialout")
    balance = args.balance or saved.get("balance", "balanced")
    held_out = args.held_out if args.held_out is not None else saved.get("held_out")
    itr = args.itr if args.itr is not None else saved.get("itr", 1)
    capture = args.capture if args.capture is not None else saved.get("capture", 1)
    append_motion = bool(saved.get("append_motion_features", False))

    ckpt_gesture_to_int: Dict[str, int] = ckpt.get("gesture_to_int", {}) or {}
    if not ckpt_gesture_to_int:
        raise RuntimeError(
            "Checkpoint has no gesture_to_int mapping; cannot run eval."
        )
    int_to_gesture = {v: k for k, v in ckpt_gesture_to_int.items()}
    default_gesture = min(ckpt_gesture_to_int.values())

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    if args.or_ambience and not args.enable_narration and not args.compare:
        print(
            "WARN: --or_ambience only applies when narration audio is built "
            "(--enable_narration or --compare); it has no effect on this run."
        )
    if (
        args.narration_backend != "template"
        and not args.enable_narration
        and not args.compare
    ):
        print(
            "WARN: --narration_backend is only used when narration audio is built "
            "(--enable_narration or --compare); template text is unused this run."
        )
    if (
        args.narration_backend == "ollama"
        and os.environ.get("COLAB_RELEASE_TAG")
    ):
        print(
            "NOTE (Google Colab): Ollama defaults to http://127.0.0.1:11434 on *this* "
            "runtime (the Colab VM). A Cloudflare tunnel only exposes web pages to "
            "your browser; it does not route urllib from this Python process to "
            "Ollama on your home machine. Options: install/run Ollama inside Colab "
            "(heavy), or use --narration_backend huggingface with HF_TOKEN / "
            "--hf_token, or keep --narration_backend template."
        )

    print(
        f"Building dataset | split={args.dataset_split} split_type={split_type} "
        f"balance={balance} held_out={held_out} itr={itr} capture={capture} "
        f"expert_only={args.expert_only}"
    )
    try:
        dataset = JIGSAWSDataset(
            data_root=args.data_root,
            split=args.dataset_split,
            split_type=split_type,
            balance=balance,
            held_out=held_out,
            itr=itr,
            expert_only=args.expert_only,
            modality="both",
            image_size=args.image_size,
            capture=capture,
            frame_stride=1,
            append_motion_features=append_motion,
        )
    except RuntimeError as e:
        raise SystemExit(
            f"Could not build test dataset: {e}\n"
            "Try a different --held_out, --itr, or --split_type."
        ) from e

    print(
        f"  trials in test split: {dataset._trial_names} "
        f"(flat_index_size={len(dataset._index)})"
    )

    # Use training-fit scaler values so generated frames match training distribution.
    scaler_params = ckpt.get("scaler_params", {}) or {}
    if "mean" in scaler_params and "scale" in scaler_params:
        dataset.scaler.mean_ = np.asarray(scaler_params["mean"], dtype=np.float64)
        dataset.scaler.scale_ = np.asarray(scaler_params["scale"], dtype=np.float64)
        print("  scaler overridden with checkpoint values")
    else:
        print("  WARNING: checkpoint has no scaler_params; using test-fit scaler")

    if args.random_trial and args.trial_name is not None:
        raise SystemExit("Use either --trial_name or --random_trial, not both.")

    if args.duration_seconds is not None:
        if args.duration_seconds <= 0:
            raise SystemExit("--duration_seconds must be positive.")
        num_out = math.ceil(args.duration_seconds * args.fps_out)
        print(
            f"  --duration_seconds={args.duration_seconds} at fps_out={args.fps_out} "
            f"-> {num_out} output frames"
        )
    else:
        num_out = args.num_frames

    if args.random_trial:
        rng = (
            random.Random(args.random_trial_seed)
            if args.random_trial_seed is not None
            else random.Random()
        )
        picked = rng.choice(dataset._trial_names)
        print(
            f"  random trial (seed={args.random_trial_seed}): {picked}"
        )
        trial_idx = _resolve_trial_index(dataset, picked)
    else:
        trial_idx = _resolve_trial_index(dataset, args.trial_name)
    trial_name = dataset._trial_names[trial_idx]
    video_path = dataset._video_paths[trial_idx]
    frame_count = dataset._frame_counts[trial_idx]
    print(f"Rendering trial: {trial_name}  (video={video_path.name}, frames={frame_count})")

    source_frames = _plan_frame_indices(
        frame_count, args.start_frame, args.frame_step, num_out
    )
    if not source_frames:
        raise SystemExit(
            f"No frames selected. start_frame={args.start_frame} "
            f"step={args.frame_step} num_frames={num_out} "
            f"but video has only {frame_count} frames."
        )
    print(
        f"  rendering {len(source_frames)} frames "
        f"[{source_frames[0]} .. {source_frames[-1]}] step={args.frame_step}"
    )

    print("Loading SD components ...")
    sampler = SDSampler(
        checkpoint_path=ckpt_path,
        device=args.device,
        image_size=args.image_size,
    )
    print(f"  running on device: {sampler.device}")

    # Open all frames from one VideoCapture handle (much faster than seeking
    # per-frame when frames are in temporal order).
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")

    W = args.image_size
    H = args.image_size
    real_writer = _open_video_writer(out_dir / "real.mp4", W, H, args.fps_out)
    gen_writer = _open_video_writer(out_dir / "generated.mp4", W, H, args.fps_out)
    side_writer: Optional[cv2.VideoWriter] = None
    if not args.no_sidebyside:
        side_writer = _open_video_writer(
            out_dir / "sidebyside.mp4", W * 2, H, args.fps_out
        )

    per_frame_records: List[Dict] = []
    total_gen_time = 0.0
    prev_gen_rgb: Optional[np.ndarray] = None
    prev_real_rgb: Optional[np.ndarray] = None
    try:
        for i, src_frame_idx in enumerate(source_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
            ret, bgr = cap.read()
            if not ret:
                print(f"  WARN: failed to read frame {src_frame_idx}; stopping")
                break
            real_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            real_rgb = cv2.resize(real_rgb, (W, H))

            kin_raw = dataset._kinematics[trial_idx][src_frame_idx]
            kin_scaled = dataset.scaler.transform(
                kin_raw.reshape(1, -1)
            ).flatten()
            label_str = dataset._frame_labels[trial_idx].get(src_frame_idx)
            if label_str is not None and label_str in ckpt_gesture_to_int:
                gesture_int = ckpt_gesture_to_int[label_str]
                gesture_source = "transcription"
            else:
                gesture_int = default_gesture
                gesture_source = "fallback"

            frame_seed = args.seed + (src_frame_idx if args.vary_seed else 0)

            # --- choose anchor image for this frame ----------------------
            init_image: Optional[np.ndarray] = None
            anchor_used = "none"
            reset_n = max(0, int(args.anchor_reset_every))
            skip_anchor = reset_n > 0 and i > 0 and (i % reset_n == 0)
            if skip_anchor:
                anchor_used = "periodic_reset"
            elif args.anchor_mode != "none" and prev_gen_rgb is not None:
                if args.anchor_mode == "prev_gen":
                    init_image = prev_gen_rgb
                    anchor_used = "prev_gen"
                elif args.anchor_mode == "prev_real":
                    init_image = prev_real_rgb
                    anchor_used = "prev_real"
                elif args.anchor_mode == "flow_warp":
                    if prev_real_rgb is None:
                        init_image = prev_gen_rgb
                        anchor_used = "prev_gen(flow_warp_fallback)"
                    else:
                        init_image = _warp_by_flow(
                            prev_gen_rgb, prev_real_rgb, real_rgb
                        )
                        anchor_used = "flow_warp"

            t0 = time.time()
            gen_img = sampler.sample(
                kin_row=kin_scaled,
                gesture_int=gesture_int,
                num_inference_steps=args.num_inference_steps,
                seed=frame_seed,
                already_scaled=True,
                init_image=init_image,
                init_strength=args.init_strength,
            )
            dt = time.time() - t0
            total_gen_time += dt

            gen_rgb = np.asarray(gen_img)
            if gen_rgb.shape[0] != H or gen_rgb.shape[1] != W:
                gen_rgb = cv2.resize(gen_rgb, (W, H))

            _write_rgb_frame(real_writer, real_rgb)
            _write_rgb_frame(gen_writer, gen_rgb)
            if side_writer is not None:
                side = np.concatenate([real_rgb, gen_rgb], axis=1)
                _write_rgb_frame(side_writer, side)

            print(
                f"  [{i + 1}/{len(source_frames)}] src_f={src_frame_idx} "
                f"gesture={label_str} (gid={gesture_int}, {gesture_source}) "
                f"anchor={anchor_used} seed={frame_seed} ({dt:.1f}s)"
            )

            per_frame_records.append(
                {
                    "output_index": i,
                    "source_frame": int(src_frame_idx),
                    "gesture": label_str,
                    "gesture_int": int(gesture_int),
                    "gesture_source": gesture_source,
                    "anchor": anchor_used,
                    "init_strength": float(args.init_strength)
                    if init_image is not None
                    else None,
                    "seed": int(frame_seed),
                    "sample_seconds": round(dt, 3),
                }
            )

            prev_gen_rgb = gen_rgb
            prev_real_rgb = real_rgb
    finally:
        cap.release()
        real_writer.release()
        gen_writer.release()
        if side_writer is not None:
            side_writer.release()

    if not per_frame_records:
        raise SystemExit(
            "No frames were written to the MP4s (files may be ~tens of bytes and "
            "unplayable). Common causes: interrupted run, first-frame read failed, "
            "or an error during the first SD sample. Check the log above and re-run."
        )

    narration_info: Dict[str, object] = {"enabled": bool(args.enable_narration)}
    narration_failure: Optional[str] = None
    if args.enable_narration:
        print("Building narration segments ...")
        expert_stats: Dict[str, Dict[str, float]] = {}
        if not args.disable_empirical_speed_stats:
            try:
                expert_stats = build_expert_speed_stats(args.data_root)
                print(
                    f"  expert speed stats loaded for {len(expert_stats)} gestures"
                )
            except Exception as e:  # pragma: no cover - defensive runtime path
                print(f"  WARN: expert speed stats unavailable: {e}")
                expert_stats = {}
        collapsed = collapse_frame_records(
            per_frame_records=per_frame_records,
            int_to_gesture=int_to_gesture,
            fps_out=args.fps_out,
        )
        narration_segments: List[Dict[str, object]] = []
        for seg in collapsed:
            src_frames = np.asarray(seg["source_frames"], dtype=np.int64)
            kin_segment = dataset._kinematics[trial_idx][src_frames]
            payload = build_narration_payload(
                gesture_label=str(seg["gesture"]),
                kinematic_segment=kin_segment,
                expert_speed_stats=expert_stats,
                min_count_for_empirical=args.narration_min_empirical_count,
            )
            narration_segments.append(
                {
                    "start_time": round(float(seg["start_time"]), 4),
                    "end_time": round(float(seg["end_time"]), 4),
                    "start_output_index": int(seg["start_output_index"]),
                    "end_output_index": int(seg["end_output_index"]),
                    "gesture": seg["gesture"],
                    "gesture_int": int(seg["gesture_int"]),
                    "source_frames": seg["source_frames"],
                    "summary": payload["summary"],
                    "narration_text": payload["narration_text"],
                    "gesture_description": payload["gesture_description"],
                    "kinematics_values": kinematics_segment_to_jsonable(kin_segment),
                }
            )
        if args.narration_backend != "template":
            print(
                f"  LLM narration ({args.narration_backend}): generating speech text, "
                "then Bark TTS ..."
            )
            narration_segments = apply_llm_narration_to_segments(
                narration_segments,
                backend=args.narration_backend,
                ollama_base_url=args.ollama_base_url,
                ollama_model=args.ollama_model,
                hf_model=args.hf_narration_model,
                hf_token=args.hf_token,
                timeout_sec=args.narration_llm_timeout_sec,
            )
        segments_path = out_dir / "narration_segments.json"
        segments_path.write_text(
            json.dumps(narration_segments, indent=2), encoding="utf-8"
        )
        print(f"  wrote {segments_path}")

        transcript_path = out_dir / "narration_transcript.txt"
        write_narration_transcript(narration_segments, transcript_path)
        print(f"  wrote {transcript_path}")

        narration_info.update(
            {
                "segments_path": str(segments_path),
                "transcript_path": str(transcript_path),
                "segment_count": len(narration_segments),
                "tts_provider": args.tts_provider,
                "tts_voice": args.tts_voice,
                "used_empirical_stats": not args.disable_empirical_speed_stats,
                "or_ambience": bool(args.or_ambience),
                "foley_dir": args.foley_dir,
                "foley_gain_db": float(args.foley_gain_db),
                "foley_align": str(args.foley_align),
                "narration_backend": args.narration_backend,
            }
        )
        if args.foley_dir and not Path(args.foley_dir).expanduser().is_dir():
            print(
                f"  WARN: --foley_dir {args.foley_dir!r} is not a directory; "
                "WAV foley will be skipped."
            )
        try:
            audio_path = out_dir / "narration_track.m4a"
            tts_info = synthesize_narration_audio(
                segments=narration_segments,
                output_audio_path=audio_path,
                provider=args.tts_provider,
                voice=args.tts_voice,
                min_segment_seconds=args.narration_min_segment_sec,
                or_ambience=args.or_ambience,
                foley_dir=args.foley_dir,
                foley_gain_db=float(args.foley_gain_db),
                foley_align=args.foley_align,
            )
            narration_info["audio_path"] = str(audio_path)
            narration_info["tts"] = tts_info

            if args.narration_default_outputs:
                generated_target = out_dir / "generated.mp4"
                generated_tmp = out_dir / "generated._tmp_narrated.mp4"
                mux_audio_to_video(
                    video_path=generated_target,
                    audio_path=audio_path,
                    output_path=generated_tmp,
                )
                generated_tmp.replace(generated_target)
                narration_info["generated_narrated_path"] = str(generated_target)
                narration_info["default_outputs_narrated"] = True
                print(f"  wrote {generated_target} (with narration)")
            else:
                generated_narrated = out_dir / "generated_narrated.mp4"
                mux_audio_to_video(
                    video_path=out_dir / "generated.mp4",
                    audio_path=audio_path,
                    output_path=generated_narrated,
                )
                narration_info["generated_narrated_path"] = str(generated_narrated)
                narration_info["default_outputs_narrated"] = False
                print(f"  wrote {generated_narrated}")

            if args.narrate_sidebyside and not args.no_sidebyside:
                if args.narration_default_outputs:
                    side_target = out_dir / "sidebyside.mp4"
                    side_tmp = out_dir / "sidebyside._tmp_narrated.mp4"
                    mux_audio_to_video(
                        video_path=side_target,
                        audio_path=audio_path,
                        output_path=side_tmp,
                    )
                    side_tmp.replace(side_target)
                    narration_info["sidebyside_narrated_path"] = str(side_target)
                    print(f"  wrote {side_target} (with narration)")
                else:
                    side_narrated = out_dir / "sidebyside_narrated.mp4"
                    mux_audio_to_video(
                        video_path=out_dir / "sidebyside.mp4",
                        audio_path=audio_path,
                        output_path=side_narrated,
                    )
                    narration_info["sidebyside_narrated_path"] = str(side_narrated)
                    print(f"  wrote {side_narrated}")
        except Exception as e:
            narration_failure = str(e)
            narration_info["error"] = narration_failure
            print(f"  WARN: narration synthesis/mux failed: {e}")

    compare_info: Dict[str, object] = {"enabled": bool(args.compare)}
    if args.compare:
        raw_root = args.raw_data_root or args.data_root
        voice_preset = args.voice_preset or args.tts_voice or DEFAULT_VOICE_PRESET
        print(
            f"[compare] generating shared Bark narration for {trial_name} "
            f"(voice_preset={voice_preset}, raw_data_root={raw_root}) ..."
        )
        try:
            wav_path = generate_shared_audio_track(
                trial_name=trial_name,
                task="suturing",
                data_root=raw_root,
                output_dir=out_dir,
                voice_preset=voice_preset,
                device=args.device,
                or_ambience=args.or_ambience,
                foley_dir=args.foley_dir,
                foley_gain_db=float(args.foley_gain_db),
                foley_align=str(args.foley_align),
                narration_backend=args.narration_backend,
                ollama_base_url=args.ollama_base_url,
                ollama_model=args.ollama_model,
                hf_narration_model=args.hf_narration_model,
                hf_token=args.hf_token,
                narration_llm_timeout_sec=args.narration_llm_timeout_sec,
            )
            print(f"[compare] wrote {wav_path}")

            raw_narrated = mux_audio_to_raw_video(
                trial_name=trial_name,
                task="suturing",
                data_root=raw_root,
                audio_path=wav_path,
                output_dir=out_dir,
            )
            print(f"[compare] wrote {raw_narrated}")

            gen_narrated = mux_audio_to_generated_video(
                generated_frames_dir=out_dir / "generated.mp4",
                audio_path=wav_path,
                trial_name=trial_name,
                output_dir=out_dir,
                fps_hint=float(args.fps_out),
            )
            print(f"[compare] wrote {gen_narrated}")

            comparison_path = side_by_side_comparison(
                raw_narrated_path=raw_narrated,
                generated_narrated_path=gen_narrated,
                output_path=out_dir / f"{trial_name}_comparison.mp4",
            )
            print(f"[compare] wrote {comparison_path}")

            shared_seg_json = out_dir / f"{trial_name}_shared_narration_segments.json"
            shared_tr_txt = out_dir / f"{trial_name}_shared_narration_transcript.txt"
            compare_info.update(
                {
                    "voice_preset": voice_preset,
                    "raw_data_root": str(raw_root),
                    "narration_wav": wav_path,
                    "raw_narrated_mp4": raw_narrated,
                    "generated_narrated_mp4": gen_narrated,
                    "comparison_mp4": comparison_path,
                    "or_ambience": bool(args.or_ambience),
                    "foley_dir": args.foley_dir,
                    "foley_gain_db": float(args.foley_gain_db),
                    "foley_align": str(args.foley_align),
                    "shared_narration_segments_json": str(shared_seg_json),
                    "shared_narration_transcript_txt": str(shared_tr_txt),
                }
            )
        except Exception as e:
            compare_info["error"] = str(e)
            print(f"[compare] WARN: failed: {e}")
            if not args.allow_narration_failure:
                raise

    metadata = {
        "checkpoint": str(ckpt_path.resolve()),
        "args": vars(args),
        "resolved": {
            "dataset_split": args.dataset_split,
            "split_type": split_type,
            "balance": balance,
            "held_out": held_out,
            "itr": itr,
            "capture": capture,
            "device": str(sampler.device),
            "model_id": sampler.model_id,
            "train_mode": sampler.train_mode,
            "cross_attn_dim": int(sampler.cross_attn_dim),
            "output_frame_target": int(num_out),
            "effective_playback_seconds": round(num_out / args.fps_out, 4),
        },
        "trial": {
            "name": trial_name,
            "video_path": str(video_path),
            "total_frames": int(frame_count),
        },
        "anchoring": {
            "mode": args.anchor_mode,
            "init_strength": float(args.init_strength),
            "anchor_reset_every": int(max(0, args.anchor_reset_every)),
        },
        "rendered_frames": len(per_frame_records),
        "total_generation_seconds": round(total_gen_time, 2),
        "avg_seconds_per_frame": (
            round(total_gen_time / len(per_frame_records), 3)
            if per_frame_records
            else None
        ),
        "gesture_to_int": ckpt_gesture_to_int,
        "narration": narration_info,
        "compare": compare_info,
        "frames": per_frame_records,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    if args.enable_narration and narration_failure and not args.allow_narration_failure:
        raise SystemExit(
            "Narration was enabled but failed; aborting to avoid silent outputs. "
            "See metadata['narration']['error'] for details, or rerun with "
            "--allow_narration_failure to keep video-only artefacts."
        )
    print(
        f"Done. {len(per_frame_records)} frames in "
        f"{total_gen_time:.1f}s "
        f"(avg {total_gen_time / max(1, len(per_frame_records)):.2f}s/frame)."
    )


if __name__ == "__main__":
    main()
