"""Overfit-one-frame diagnostic.

Train a **fresh** LoRA + ``KinematicEncoder`` on a single JIGSAWS
``(frame, kinematics, gesture)`` tuple for a short number of steps,
snapshotting generation at several checkpoints.

Why: if the training loop cannot memorise a single frame, the issue is
with the loop / encoder / wiring, not with the amount of data.  If it
can memorise one frame, the pipeline is structurally sound and the
"messy output" problem is about data scale / training steps / resolution.

Output artefacts (under ``outputs/diagnostics/overfit_<run_name>/``):

* ``progression_grid.png`` — first column real frame, subsequent columns
  generated frames at each snapshot step.
* ``loss_curve.png`` — MSE loss per step.
* ``losses.json`` — raw per-step loss values.
* ``metadata.json`` — args, chosen trial/frame, snapshot steps, final
  loss, final generated PSNR vs real.

Example::

    python scripts/overfit_one_frame.py \\
        --data_root /Users/amyngo/SS-SD/data/gdrive_cache \\
        --steps 500 --snapshot_steps 50,100,250,500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.data.jigsaws_dataset import JIGSAWSDataset
from suturing_pipeline.synthesis.kinematic_encoder import (
    KinematicEncoder,
    encode_clip_scene_embedding,
)
from suturing_pipeline.synthesis.sd_sampler import resolve_device


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overfit the SD LoRA + KinematicEncoder to one frame."
    )
    p.add_argument("--data_root", required=True)
    p.add_argument(
        "--model_id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    p.add_argument("--output_dir", default="./outputs/diagnostics")
    p.add_argument("--run_name", default=None)

    # Which frame to overfit
    p.add_argument(
        "--trial",
        default=None,
        help="JIGSAWS trial name (e.g. 'Suturing_B001').  "
        "Default: first trial in the chosen split.",
    )
    p.add_argument(
        "--frame_idx",
        type=int,
        default=None,
        help="Frame index within the trial.  Default: middle frame.",
    )

    # Dataset plumbing (the dataset is only used to look up one frame,
    # so these mainly affect path resolution).
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--split_type", default="onetrialout")
    p.add_argument("--balance", default="balanced")
    p.add_argument("--held_out", type=int, default=None)
    p.add_argument("--itr", type=int, default=1)
    p.add_argument(
        "--expert_only",
        action="store_true",
        help="Self-reported expert only (meta file column 2). See jigsaws_metafile_layout.",
    )
    p.add_argument("--capture", type=int, default=1, choices=[1, 2])
    p.add_argument("--frame_stride", type=int, default=30)

    # Training
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)

    # Snapshots
    p.add_argument(
        "--snapshot_steps",
        default="50,100,250,500",
        help="Comma-separated step numbers at which to run DDIM inference.",
    )
    p.add_argument("--snapshot_ddim_steps", type=int, default=25)
    p.add_argument("--snapshot_seed", type=int, default=0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--append_motion_features",
        action="store_true",
        help="Match train_sd.py: append velocity/accel/jerk scalars to kin.",
    )
    p.add_argument("--num_semantic_tokens", type=int, default=0)
    p.add_argument("--clip_scene_prompt", default=None)
    return p.parse_args()


def _parse_snapshot_steps(arg: str, max_step: int) -> List[int]:
    steps = sorted({int(s) for s in arg.split(",") if s.strip()})
    steps = [s for s in steps if 1 <= s <= max_step]
    if max_step not in steps:
        steps.append(max_step)
    return sorted(set(steps))


def _tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    img = (frame.clamp(-1, 1) + 1) / 2 * 255
    return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse))


@torch.no_grad()
def _run_ddim(
    vae,
    unet,
    encoder: KinematicEncoder,
    kin_scaled: torch.Tensor,
    gesture_int: int,
    image_size: int,
    num_steps: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Inline DDIM inference using the current (training) weights."""
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_pretrained(
        _run_ddim.model_id, subfolder="scheduler"  # type: ignore[attr-defined]
    )
    scheduler.set_timesteps(num_steps, device=device)

    encoder.eval()
    unet.eval()
    try:
        gesture_tensor = torch.tensor([int(gesture_int)], dtype=torch.long, device=device)
        cond = encoder(kin_scaled.unsqueeze(0).to(device), gesture_tensor)

        in_channels = unet.config.in_channels if hasattr(unet, "config") else 4
        latents = torch.randn(
            1,
            in_channels,
            image_size // 8,
            image_size // 8,
            device=device,
            generator=torch.Generator(device=device).manual_seed(int(seed)),
        )
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, t)
            noise_pred = unet(latent_input, t, encoder_hidden_states=cond).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        vae_scale = vae.config.scaling_factor
        image = vae.decode(latents / vae_scale).sample
    finally:
        encoder.train()
        unet.train()

    image = (image.clamp(-1, 1) + 1) / 2 * 255
    return image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _compose_progression_grid(
    real: np.ndarray,
    snapshots: List[Dict],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    k = len(snapshots)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.6), squeeze=False)
    axes[0, 0].imshow(real)
    axes[0, 0].set_title("REAL target", fontsize=10)
    axes[0, 0].axis("off")
    for i, snap in enumerate(snapshots, start=1):
        axes[0, i].imshow(snap["image"])
        axes[0, i].set_title(
            f"step {snap['step']}\nPSNR={snap['psnr']:.2f} dB",
            fontsize=10,
        )
        axes[0, i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _compose_loss_curve(losses: List[float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (noise prediction)")
    ax.set_title("Overfit-one-frame loss curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"overfit_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset (used only to look up one frame).
    # ------------------------------------------------------------------
    print("Building dataset ...")
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
        append_motion_features=args.append_motion_features,
    )
    print(
        f"  trials={len(dataset._trial_names)} index_size={len(dataset._index)}"
    )

    # Pick the trial.
    if args.trial is None:
        trial_idx = 0
    else:
        if args.trial not in dataset._trial_names:
            raise SystemExit(
                f"Trial '{args.trial}' not found.  Available: "
                f"{dataset._trial_names}"
            )
        trial_idx = dataset._trial_names.index(args.trial)
    trial_name = dataset._trial_names[trial_idx]
    frame_count = dataset._frame_counts[trial_idx]

    # Pick the frame.
    if args.frame_idx is None:
        target_frame_idx = frame_count // 2
    else:
        target_frame_idx = args.frame_idx
    target_frame_idx = max(0, min(target_frame_idx, frame_count - 1))

    # Find a flat index that matches (or, if the stride skipped it, the
    # closest one).  We'll then override the returned kinematics row to
    # exactly match target_frame_idx for precise labelling.
    flat_idx = None
    best = None
    for fi, (ti, fidx) in enumerate(dataset._index):
        if ti != trial_idx:
            continue
        if fidx == target_frame_idx:
            flat_idx = fi
            break
        d = abs(fidx - target_frame_idx)
        if best is None or d < best[0]:
            best = (d, fi, fidx)
    if flat_idx is None:
        assert best is not None
        flat_idx = best[1]
        target_frame_idx = best[2]
        print(
            f"  requested frame not in stride grid; using nearest: "
            f"frame_idx={target_frame_idx}"
        )

    print(f"Target: trial={trial_name} frame_idx={target_frame_idx}")

    frame_tensor, kin_tensor, gesture_int = dataset[flat_idx]
    label_str = dataset._frame_labels[trial_idx].get(target_frame_idx, "<none>")
    print(
        f"  gesture='{label_str}' gesture_int={gesture_int} "
        f"num_gestures_in_dataset={dataset.num_gestures}"
    )
    real_uint8 = _tensor_to_uint8(frame_tensor)

    # ------------------------------------------------------------------
    # Build SD components.
    # ------------------------------------------------------------------
    print(f"Loading SD components from {args.model_id} ...")
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model

    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(
        device
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    vae.requires_grad_(False)
    vae.eval()

    unet.requires_grad_(False)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_k", "to_v", "to_q", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.train()

    cross_attn_dim = unet.config.cross_attention_dim if hasattr(unet, "config") else 768

    kin_dim = dataset._kinematics[trial_idx].shape[1]
    clip_feat = None
    if args.clip_scene_prompt:
        clip_feat = encode_clip_scene_embedding(
            args.model_id,
            args.clip_scene_prompt,
            device=torch.device("cpu"),
        )

    encoder = KinematicEncoder(
        kin_dim=kin_dim,
        num_gestures=max(dataset.num_gestures, 1),
        seq_len=77,
        embed_dim=cross_attn_dim,
        num_semantic_tokens=args.num_semantic_tokens,
        clip_scene_feature=clip_feat,
    ).to(device)
    encoder.train()

    # Stash model_id on _run_ddim so it can rebuild the DDIMScheduler.
    _run_ddim.model_id = args.model_id  # type: ignore[attr-defined]

    trainable_params = list(encoder.parameters()) + [
        p for p in unet.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Single-sample batch, pre-moved to device.
    frame_b = frame_tensor.unsqueeze(0).to(device)
    kin_b = kin_tensor.unsqueeze(0).to(device)
    gesture_b = torch.tensor([int(gesture_int)], dtype=torch.long, device=device)
    vae_scale = vae.config.scaling_factor

    snapshot_steps = _parse_snapshot_steps(args.snapshot_steps, args.steps)
    print(f"Snapshots at steps: {snapshot_steps}")

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    print(f"Starting overfit for {args.steps} steps ...")
    losses: List[float] = []
    snapshots: List[Dict] = []

    t0 = time.time()
    for step in range(1, args.steps + 1):
        with torch.no_grad():
            latents = vae.encode(frame_b).latent_dist.sample() * vae_scale

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        cond = encoder(kin_b, gesture_b)
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=cond).sample
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

        if step % max(1, args.steps // 20) == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step {step:>5d}/{args.steps}  loss={loss_val:.5f}  ({elapsed:.1f}s)")

        if step in snapshot_steps:
            print(f"  -> snapshot DDIM inference at step {step} ...")
            s0 = time.time()
            gen = _run_ddim(
                vae=vae,
                unet=unet,
                encoder=encoder,
                kin_scaled=kin_tensor,
                gesture_int=int(gesture_int),
                image_size=args.image_size,
                num_steps=args.snapshot_ddim_steps,
                seed=args.snapshot_seed,
                device=device,
            )
            psnr = _psnr(real_uint8, gen)
            snapshots.append(
                {
                    "step": int(step),
                    "image": gen,
                    "psnr": float(psnr),
                    "seconds": float(time.time() - s0),
                }
            )
            print(f"     PSNR vs real = {psnr:.2f} dB   ({time.time() - s0:.1f}s)")

    total_t = time.time() - t0
    print(f"Overfit done in {total_t:.1f}s (avg {total_t / max(1, args.steps):.3f}s/step)")

    # ------------------------------------------------------------------
    # Artefacts.
    # ------------------------------------------------------------------
    grid_path = out_dir / "progression_grid.png"
    _compose_progression_grid(real_uint8, snapshots, grid_path)
    print(f"Wrote {grid_path}")

    loss_path = out_dir / "loss_curve.png"
    _compose_loss_curve(losses, loss_path)
    print(f"Wrote {loss_path}")

    (out_dir / "losses.json").write_text(
        json.dumps({"losses": losses}, indent=2), encoding="utf-8"
    )

    metadata = {
        "args": vars(args),
        "resolved": {
            "device": str(device),
            "trial": trial_name,
            "frame_idx": int(target_frame_idx),
            "gesture": label_str,
            "gesture_int": int(gesture_int),
            "num_gestures_in_dataset": int(dataset.num_gestures),
            "image_size": int(args.image_size),
            "lora_rank": int(args.lora_rank),
            "cross_attn_dim": int(cross_attn_dim),
            "total_seconds": float(total_t),
        },
        "losses_summary": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "min": min(losses) if losses else None,
            "mean_last_10pct": float(np.mean(losses[-max(1, len(losses) // 10):])) if losses else None,
        },
        "snapshots": [
            {k: v for k, v in s.items() if k != "image"} for s in snapshots
        ],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'metadata.json'}")

    final_psnr = snapshots[-1]["psnr"] if snapshots else float("nan")
    print()
    print("Interpretation guide:")
    print(f"  Final-step PSNR vs real = {final_psnr:.2f} dB")
    print("  > 25 dB  : loop can memorise a single frame; data/steps/LoRA-rank")
    print("              are the bottleneck, not the architecture.")
    print("  18 - 25  : partial memorisation; try more steps, higher LoRA rank,")
    print("              or check the VAE reconstruction diagnostic.")
    print("  < 18     : the training loop or conditioning wiring is suspect.")


if __name__ == "__main__":
    main()
