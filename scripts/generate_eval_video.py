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
the most coherent output under these constraints.

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
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

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
        help="Restrict to expert trials (metafile skill == Expert).",
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

    ckpt_gesture_to_int: Dict[str, int] = ckpt.get("gesture_to_int", {}) or {}
    if not ckpt_gesture_to_int:
        raise RuntimeError(
            "Checkpoint has no gesture_to_int mapping; cannot run eval."
        )
    default_gesture = min(ckpt_gesture_to_int.values())

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

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
            if args.anchor_mode != "none" and prev_gen_rgb is not None:
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
        },
        "rendered_frames": len(per_frame_records),
        "total_generation_seconds": round(total_gen_time, 2),
        "avg_seconds_per_frame": (
            round(total_gen_time / len(per_frame_records), 3)
            if per_frame_records
            else None
        ),
        "gesture_to_int": ckpt_gesture_to_int,
        "frames": per_frame_records,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    print(
        f"Done. {len(per_frame_records)} frames in "
        f"{total_gen_time:.1f}s "
        f"(avg {total_gen_time / max(1, len(per_frame_records)):.2f}s/frame)."
    )


if __name__ == "__main__":
    main()
