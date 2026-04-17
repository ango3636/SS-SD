"""Qualitative evaluation grid for a trained kinematic-conditioned SD model.

Loads a checkpoint produced by ``scripts/train_sd.py``, iterates over the
matching held-out JIGSAWS test split, and writes:

* ``eval_grid.png`` — N rows, 2 columns (real frame | generated frame).
* ``gesture_sweep.png`` — 1 row, K columns.  Kinematics fixed, gesture
  label varied across the checkpoint's ``gesture_to_int`` mapping.
* ``metadata.json`` — run args, checkpoint path, per-sample records.

Example::

    python scripts/generate_eval_grid.py \\
        --checkpoint checkpoints/suturing_expert_lora/step_1480.pt \\
        --data_root  /Users/amyngo/SS-SD/data/gdrive_cache
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.data.jigsaws_dataset import JIGSAWSDataset
from suturing_pipeline.synthesis.sd_sampler import SDSampler


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate real-vs-generated grid + gesture sweep."
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
    p.add_argument("--output_dir", default="./outputs/eval")

    # Split selection — defaults read from checkpoint if omitted.
    p.add_argument("--split_type", default=None)
    p.add_argument("--balance", default=None)
    p.add_argument("--held_out", type=int, default=None)
    p.add_argument("--itr", type=int, default=None)
    p.add_argument("--capture", type=int, default=None, choices=[1, 2])
    p.add_argument(
        "--expert_only",
        action="store_true",
        help="Restrict test split to expert trials.  By default the test "
        "split includes all skill levels present in the fold.",
    )

    # Sampling
    p.add_argument("--num_samples", type=int, default=12)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--sweep_steps", type=int, default=25)
    p.add_argument(
        "--frame_stride",
        type=int,
        default=30,
        help="Stride used when building the test dataset index "
        "(not the same as training stride).",
    )
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--skip_sweep",
        action="store_true",
        help="Skip the gesture_sweep.png artefact (faster smoke tests).",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Sub-folder under --output_dir.  Defaults to a timestamp.",
    )
    return p.parse_args()


def _build_gesture_index_map(
    dataset: JIGSAWSDataset,
    ckpt_gesture_to_int: Dict[str, int],
) -> Tuple[Dict[str, List[int]], int]:
    """Walk ``dataset._index`` and group flat indices by gesture label.

    Indices whose transcription gesture is not in the checkpoint's
    ``gesture_to_int`` mapping are dropped (the model never saw them).

    Returns
    -------
    gesture_to_indices:
        Map from gesture label string -> list of flat dataset indices.
    skipped:
        Number of dataset entries skipped because their gesture was
        unknown to the checkpoint.
    """
    gesture_to_indices: Dict[str, List[int]] = {
        label: [] for label in ckpt_gesture_to_int
    }
    skipped = 0
    for flat_idx, (trial_idx, frame_idx) in enumerate(dataset._index):
        label = dataset._frame_labels[trial_idx].get(frame_idx)
        if label is None or label not in ckpt_gesture_to_int:
            skipped += 1
            continue
        gesture_to_indices[label].append(flat_idx)
    return gesture_to_indices, skipped


def _balanced_sample(
    gesture_to_indices: Dict[str, List[int]],
    num_samples: int,
    rng: random.Random,
) -> List[Tuple[str, int]]:
    """Pick ``num_samples`` flat indices, round-robin over gestures.

    Returns a list of ``(gesture_label, flat_idx)`` tuples.
    """
    pools = {
        g: rng.sample(idxs, len(idxs))
        for g, idxs in gesture_to_indices.items()
        if idxs
    }
    order = sorted(pools.keys())
    picked: List[Tuple[str, int]] = []
    while pools and len(picked) < num_samples:
        for g in list(order):
            if g not in pools:
                continue
            if not pools[g]:
                pools.pop(g)
                continue
            picked.append((g, pools[g].pop()))
            if len(picked) >= num_samples:
                break
    return picked


def _tensor_to_uint8(frame_tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``[3, H, W]`` frame tensor in ``[-1, 1]`` to HWC uint8."""
    img = (frame_tensor.clamp(-1, 1) + 1) / 2 * 255
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _compose_real_vs_generated_grid(
    pairs: List[Dict],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n = len(pairs)
    fig, axes = plt.subplots(
        n, 2, figsize=(6, 3 * n), squeeze=False
    )
    for row, record in enumerate(pairs):
        axes[row, 0].imshow(record["real"])
        axes[row, 0].set_title(
            f"REAL | {record['trial']} | f={record['frame_idx']} | {record['gesture']}",
            fontsize=9,
        )
        axes[row, 0].axis("off")
        axes[row, 1].imshow(record["generated"])
        axes[row, 1].set_title(
            f"GENERATED | {record['gesture']} (gid={record['gesture_int']})",
            fontsize=9,
        )
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _compose_gesture_sweep(
    sweep_records: List[Dict],
    ref_record: Dict,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    k = len(sweep_records)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.5), squeeze=False)
    axes[0, 0].imshow(ref_record["real"])
    axes[0, 0].set_title(
        f"REAL ref\n{ref_record['trial']} f={ref_record['frame_idx']}\n"
        f"(actual: {ref_record['gesture']})",
        fontsize=9,
    )
    axes[0, 0].axis("off")
    for i, rec in enumerate(sweep_records, start=1):
        axes[0, i].imshow(rec["generated"])
        axes[0, i].set_title(
            f"GEN | {rec['gesture']} (gid={rec['gesture_int']})",
            fontsize=9,
        )
        axes[0, i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Load checkpoint first so we can pull saved training args.
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved: Dict = ckpt.get("args", {}) or {}

    # Resolve split / capture defaults from the training run if unspecified.
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
    print(
        f"Checkpoint gesture vocabulary ({len(ckpt_gesture_to_int)}): "
        f"{sorted(ckpt_gesture_to_int, key=ckpt_gesture_to_int.get)}"
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ------------------------------------------------------------------
    # Build test dataset (same fold as training by default).
    # ------------------------------------------------------------------
    print(
        f"Building test dataset | split_type={split_type} balance={balance} "
        f"held_out={held_out} itr={itr} capture={capture} "
        f"expert_only={args.expert_only}"
    )
    try:
        dataset = JIGSAWSDataset(
            data_root=args.data_root,
            split="test",
            split_type=split_type,
            balance=balance,
            held_out=held_out,
            itr=itr,
            expert_only=args.expert_only,
            modality="both",
            image_size=args.image_size,
            capture=capture,
            frame_stride=args.frame_stride,
        )
    except RuntimeError as e:
        raise SystemExit(
            f"Could not build test dataset: {e}\n"
            "Try a different --held_out, --itr, or --split_type "
            "(e.g. 'userout'), or drop --expert_only."
        ) from e

    print(
        f"  trials={len(dataset._trial_names)} | "
        f"flat_index_size={len(dataset._index)}"
    )

    # Override the test-split scaler with the training-split scaler from ckpt.
    scaler_params = ckpt.get("scaler_params", {}) or {}
    if "mean" in scaler_params and "scale" in scaler_params:
        dataset.scaler.mean_ = np.asarray(scaler_params["mean"], dtype=np.float64)
        dataset.scaler.scale_ = np.asarray(scaler_params["scale"], dtype=np.float64)
        print("  scaler overridden with checkpoint values")
    else:
        print(
            "  WARNING: checkpoint has no scaler_params; "
            "using test-split-fit scaler (may differ from training)."
        )

    # ------------------------------------------------------------------
    # Choose samples balanced across gestures the model knows about.
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    gesture_to_indices, skipped = _build_gesture_index_map(
        dataset, ckpt_gesture_to_int
    )
    total_usable = sum(len(v) for v in gesture_to_indices.values())
    print(
        f"  usable frames: {total_usable} "
        f"(skipped {skipped} with gestures outside ckpt vocab)"
    )
    if total_usable == 0:
        raise SystemExit(
            "No test frames with gesture labels in the checkpoint vocabulary."
        )

    picks = _balanced_sample(gesture_to_indices, args.num_samples, rng)
    print(f"  selected {len(picks)} frames for eval grid")
    per_gesture_counts: Dict[str, int] = {}
    for g, _ in picks:
        per_gesture_counts[g] = per_gesture_counts.get(g, 0) + 1
    print(f"  per-gesture counts: {per_gesture_counts}")

    # ------------------------------------------------------------------
    # Load SD once.
    # ------------------------------------------------------------------
    print("Loading SD components ...")
    sampler = SDSampler(
        checkpoint_path=ckpt_path,
        device=args.device,
        image_size=args.image_size,
    )
    print(f"  running on device: {sampler.device}")

    # ------------------------------------------------------------------
    # Generate pairs.
    # ------------------------------------------------------------------
    pairs: List[Dict] = []
    for i, (label, flat_idx) in enumerate(picks):
        trial_idx, frame_idx = dataset._index[flat_idx]
        trial_name = dataset._trial_names[trial_idx]
        gesture_int = ckpt_gesture_to_int[label]

        frame_tensor, kin_tensor, _ = dataset[flat_idx]
        real_np = _tensor_to_uint8(frame_tensor)

        kin_scaled = kin_tensor.cpu().numpy().astype(np.float64)

        t0 = time.time()
        gen_img = sampler.sample(
            kin_row=kin_scaled,
            gesture_int=gesture_int,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + i,
            already_scaled=True,
        )
        dt = time.time() - t0
        print(
            f"  [{i + 1}/{len(picks)}] {trial_name} f={frame_idx} "
            f"gesture={label} ({dt:.1f}s)"
        )

        pairs.append(
            {
                "trial": trial_name,
                "frame_idx": int(frame_idx),
                "gesture": label,
                "gesture_int": int(gesture_int),
                "flat_idx": int(flat_idx),
                "seed": int(args.seed + i),
                "real": real_np,
                "generated": np.asarray(gen_img),
            }
        )

    # ------------------------------------------------------------------
    # Compose eval grid.
    # ------------------------------------------------------------------
    grid_path = out_dir / "eval_grid.png"
    print(f"Writing {grid_path} ...")
    _compose_real_vs_generated_grid(pairs, grid_path)

    # ------------------------------------------------------------------
    # Gesture sweep.
    # ------------------------------------------------------------------
    sweep_info: Optional[Dict] = None
    if not args.skip_sweep and pairs:
        ref = pairs[0]
        ref_trial_idx, ref_frame_idx = dataset._index[ref["flat_idx"]]
        ref_kin_raw = dataset._kinematics[ref_trial_idx][ref_frame_idx]
        print(
            f"Gesture sweep on {ref['trial']} f={ref_frame_idx} "
            f"across {len(ckpt_gesture_to_int)} gestures ..."
        )
        sweep_records: List[Dict] = []
        for j, (label, gid) in enumerate(
            sorted(ckpt_gesture_to_int.items(), key=lambda kv: kv[1])
        ):
            t0 = time.time()
            gen_img = sampler.sample(
                kin_row=ref_kin_raw,
                gesture_int=gid,
                num_inference_steps=args.sweep_steps,
                seed=args.seed + 1000 + j,
            )
            dt = time.time() - t0
            print(f"  gesture {label} (gid={gid}) done ({dt:.1f}s)")
            sweep_records.append(
                {
                    "gesture": label,
                    "gesture_int": int(gid),
                    "seed": int(args.seed + 1000 + j),
                    "generated": np.asarray(gen_img),
                }
            )
        sweep_path = out_dir / "gesture_sweep.png"
        print(f"Writing {sweep_path} ...")
        _compose_gesture_sweep(sweep_records, ref, sweep_path)
        sweep_info = {
            "ref_trial": ref["trial"],
            "ref_frame_idx": int(ref_frame_idx),
            "ref_actual_gesture": ref["gesture"],
            "gestures": [
                {
                    "gesture": r["gesture"],
                    "gesture_int": r["gesture_int"],
                    "seed": r["seed"],
                }
                for r in sweep_records
            ],
        }

    # ------------------------------------------------------------------
    # metadata.json
    # ------------------------------------------------------------------
    metadata = {
        "checkpoint": str(ckpt_path.resolve()),
        "args": vars(args),
        "resolved": {
            "split_type": split_type,
            "balance": balance,
            "held_out": held_out,
            "itr": itr,
            "capture": capture,
            "device": str(sampler.device),
            "model_id": sampler.model_id,
            "train_mode": sampler.train_mode,
            "cross_attn_dim": int(sampler.cross_attn_dim),
        },
        "gesture_to_int": ckpt_gesture_to_int,
        "test_trials": dataset._trial_names,
        "test_flat_index_size": len(dataset._index),
        "skipped_unknown_gesture": int(skipped),
        "per_gesture_sampled": per_gesture_counts,
        "samples": [
            {
                "trial": r["trial"],
                "frame_idx": r["frame_idx"],
                "gesture": r["gesture"],
                "gesture_int": r["gesture_int"],
                "flat_idx": r["flat_idx"],
                "seed": r["seed"],
            }
            for r in pairs
        ],
        "gesture_sweep": sweep_info,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
