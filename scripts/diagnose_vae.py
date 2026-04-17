"""VAE reconstruction diagnostic.

Run a handful of real JIGSAWS Suturing frames through Stable Diffusion
1.5's frozen VAE (encode -> decode) and compare against the originals.

This tells us the **ceiling** of the kinematic-conditioned SD pipeline:
if the VAE itself cannot reconstruct surgical frames faithfully, no
amount of LoRA or U-Net fine-tuning will produce clean outputs — the
VAE would have to be fine-tuned on surgical frames as well.

Output artefacts (under ``outputs/diagnostics/vae_<run_name>/``):

* ``vae_recon_grid.png`` — N rows x 2 columns (real | reconstruction),
  captioned with per-image PSNR.
* ``metadata.json`` — args, resolution, per-sample PSNR / MSE, mean PSNR.

Example::

    python scripts/diagnose_vae.py \\
        --data_root /Users/amyngo/SS-SD/data/gdrive_cache \\
        --num_samples 6 --image_size 256
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.data.jigsaws_dataset import JIGSAWSDataset
from suturing_pipeline.synthesis.sd_sampler import resolve_device


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose VAE reconstruction quality on JIGSAWS frames."
    )
    p.add_argument(
        "--data_root", required=True, help="JIGSAWS root (same as training)."
    )
    p.add_argument(
        "--model_id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HuggingFace SD model id whose VAE will be tested.",
    )
    p.add_argument("--output_dir", default="./outputs/diagnostics")
    p.add_argument("--run_name", default=None)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which JIGSAWS split to pull sample frames from.",
    )
    p.add_argument("--split_type", default="onetrialout")
    p.add_argument("--balance", default="balanced")
    p.add_argument("--held_out", type=int, default=None)
    p.add_argument("--itr", type=int, default=1)
    p.add_argument("--expert_only", action="store_true")
    p.add_argument("--capture", type=int, default=1, choices=[1, 2])
    p.add_argument("--frame_stride", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def _tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    img = (frame.clamp(-1, 1) + 1) / 2 * 255
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _compose_grid(records: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
    for row, rec in enumerate(records):
        axes[row, 0].imshow(rec["real"])
        axes[row, 0].set_title(
            f"REAL | {rec['trial']} f={rec['frame_idx']}", fontsize=9
        )
        axes[row, 0].axis("off")
        axes[row, 1].imshow(rec["recon"])
        axes[row, 1].set_title(
            f"VAE RECON | PSNR={rec['psnr']:.2f} dB  MSE={rec['mse']:.1f}",
            fontsize=9,
        )
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    run_name = args.run_name or f"{time.strftime('%Y%m%d_%H%M%S')}_sz{args.image_size}"
    out_dir = Path(args.output_dir) / f"vae_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    # Build dataset purely for its frame-lookup machinery.
    print(
        f"Building dataset | split={args.split} image_size={args.image_size} "
        f"frame_stride={args.frame_stride} expert_only={args.expert_only}"
    )
    dataset = JIGSAWSDataset(
        data_root=args.data_root,
        split=args.split,
        split_type=args.split_type,
        balance=args.balance,
        held_out=args.held_out,
        itr=args.itr,
        expert_only=args.expert_only,
        modality="both",
        image_size=args.image_size,
        capture=args.capture,
        frame_stride=args.frame_stride,
    )
    print(
        f"  trials={len(dataset._trial_names)} index_size={len(dataset._index)}"
    )
    if len(dataset._index) == 0:
        raise SystemExit("Empty dataset; nothing to reconstruct.")

    # Pick evenly-spaced indices so samples span the trial(s) rather than
    # clustering at the start.
    rng = random.Random(args.seed)
    n = min(args.num_samples, len(dataset._index))
    step = max(1, len(dataset._index) // n)
    candidate_idxs = list(range(0, len(dataset._index), step))[:n]
    rng.shuffle(candidate_idxs)

    # Load VAE.
    from diffusers import AutoencoderKL

    print(f"Loading VAE from {args.model_id} ...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.eval()
    vae.requires_grad_(False)
    vae_scale = vae.config.scaling_factor

    records: List[Dict] = []
    psnrs: List[float] = []
    mses: List[float] = []

    for i, flat_idx in enumerate(candidate_idxs):
        trial_idx, frame_idx = dataset._index[flat_idx]
        trial = dataset._trial_names[trial_idx]
        frame_tensor, _, _ = dataset[flat_idx]

        real_uint8 = _tensor_to_uint8(frame_tensor)

        with torch.no_grad():
            x = frame_tensor.unsqueeze(0).to(device)
            latent = vae.encode(x).latent_dist.sample() * vae_scale
            recon = vae.decode(latent / vae_scale).sample
        recon_uint8 = _tensor_to_uint8(recon[0])

        psnr = _psnr(real_uint8, recon_uint8)
        mse = float(np.mean((real_uint8.astype(np.float64) - recon_uint8.astype(np.float64)) ** 2))
        psnrs.append(psnr)
        mses.append(mse)

        print(
            f"  [{i + 1}/{n}] {trial} f={frame_idx}  PSNR={psnr:.2f} dB  "
            f"MSE={mse:.1f}"
        )

        records.append(
            {
                "trial": trial,
                "frame_idx": int(frame_idx),
                "flat_idx": int(flat_idx),
                "psnr": float(psnr),
                "mse": float(mse),
                "real": real_uint8,
                "recon": recon_uint8,
            }
        )

    grid_path = out_dir / "vae_recon_grid.png"
    print(f"Writing {grid_path} ...")
    _compose_grid(records, grid_path)

    mean_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    mean_mse = float(np.mean(mses)) if mses else float("nan")
    print(f"Mean PSNR over {n} samples: {mean_psnr:.2f} dB (mean MSE {mean_mse:.1f})")

    metadata = {
        "args": vars(args),
        "resolved": {
            "device": str(device),
            "vae_scaling_factor": float(vae_scale),
            "model_id": args.model_id,
        },
        "mean_psnr_db": mean_psnr,
        "mean_mse": mean_mse,
        "samples": [
            {k: v for k, v in rec.items() if k not in ("real", "recon")}
            for rec in records
        ],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Done.")
    print()
    print("Interpretation guide:")
    print("  > 32 dB         near-perfect reconstruction; VAE is NOT the bottleneck.")
    print("  27 - 32 dB      good; small blur but fine-grained detail preserved.")
    print("  22 - 27 dB      noticeable blur / colour shift; VAE is a modest ceiling.")
    print("  < 22 dB         VAE cannot represent surgical frames well; consider")
    print("                   running at --image_size 512 or fine-tuning the VAE.")


if __name__ == "__main__":
    main()
