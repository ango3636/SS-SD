"""Train a Stable Diffusion model conditioned on JIGSAWS Suturing kinematics.

The CLIP text encoder is replaced by a learned
:class:`~suturing_pipeline.synthesis.kinematic_encoder.KinematicEncoder`
that projects kinematic vectors + gesture labels into the U-Net's
cross-attention space.  Optional **learnable semantic tokens** and a
**CLIP-derived scene token** (from ``--clip_scene_prompt``) extend conditioning
for non-kinematic scene content.

Only the JIGSAWS ``suturing`` task is supported — kinematics for the other
JIGSAWS tasks are not available in this project.

Supports two training modes:

* **lora** — freeze the full U-Net and attach LoRA adapters to the
  cross-attention projection layers (recommended, ~1 % trainable params).
* **full** — unfreeze all cross-attention ``to_k`` / ``to_v`` projections
  while keeping the rest of the U-Net frozen.

Optional **temporal delta loss** (``--temporal_delta_weight``): each batch item
is two frames from the same trial; we add λ·L1(pred Δ latent − clean Δ latent)
at the same DDPM timestep to encourage motion-consistent latents across time.

Example::

    python scripts/train_sd.py \\
        --data_root /data/JIGSAWS \\
        --expert_only \\
        --train_mode lora \\
        --batch_size 4 \\
        --lr 1e-4 \\
        --epochs 50 \\
        --save_dir ./checkpoints/suturing_expert_lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.data.jigsaws_dataset import JIGSAWSDataset
from suturing_pipeline.synthesis.kinematic_encoder import (
    KinematicEncoder,
    encode_clip_scene_embedding,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train kinematic-conditioned Stable Diffusion."
    )
    p.add_argument("--data_root", required=True, help="Path to JIGSAWS root.")
    p.add_argument("--split_type", default="onetrialout")
    p.add_argument("--balance", default="balanced")
    p.add_argument(
        "--held_out",
        type=int,
        default=None,
        help="Which trial group to hold out (e.g. 10 for 10_Out). "
        "If omitted, the first available fold is used.",
    )
    p.add_argument("--itr", type=int, default=1)
    p.add_argument(
        "--expert_only",
        action="store_true",
        help="Keep only trials with self-reported expert (meta file col 2, 1-based N/I/E).",
    )
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--capture", type=int, default=1, choices=[1, 2])
    p.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Sample every N-th frame. Use 30 for ~1 fps on CPU.",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--model_id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HuggingFace model ID for the base SD checkpoint.",
    )
    p.add_argument(
        "--train_mode",
        choices=["lora", "full"],
        default="lora",
        help="'lora' attaches LoRA to cross-attn layers; "
        "'full' unfreezes cross-attn to_k/to_v projections.",
    )
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save a checkpoint every N global steps.",
    )
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--device",
        default=None,
        help="Override device (e.g. 'cuda', 'mps', 'cpu').",
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--append_motion_features",
        action="store_true",
        help="Append per-frame velocity / acceleration / jerk (4 values) to each "
        "raw 76-dim kinematics row (80-dim input).",
    )
    p.add_argument(
        "--num_semantic_tokens",
        type=int,
        default=0,
        help="Learnable soft-prompt tokens concatenated to cross-attention for "
        "non-kinematic scene context (0 = disabled).",
    )
    p.add_argument(
        "--clip_scene_prompt",
        default=None,
        help="Optional. Frozen CLIP text-encoder embedding of this string; one "
        "projected token is concatenated after kinematic + semantic tokens. "
        "Example: 'robotic laparoscopic suturing with needle and suture thread'.",
    )
    p.add_argument(
        "--temporal_delta_weight",
        type=float,
        default=0.0,
        help="If > 0, each batch item is two frames from the same trial; adds "
        "λ · L1(pred Δ latent − clean Δ latent) alongside noise MSE (same DDPM "
        "timestep per pair). Use ~0.05–0.5 to start; reduces long-clip drift.",
    )
    p.add_argument(
        "--temporal_pair_gap",
        type=int,
        default=1,
        help="Frame-index gap between paired frames when --temporal_delta_weight "
        "> 0 (1 = consecutive frames).",
    )
    return p.parse_args()


def _predict_original_from_epsilon(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """Map epsilon prediction to predicted clean latent x₀ (DDPM, epsilon loss)."""
    ac = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )[timesteps]
    ac = ac.reshape(-1, 1, 1, 1).clamp(min=1e-9)
    sqrt_ac = ac.sqrt()
    sqrt_one_minus = (1.0 - ac).sqrt()
    return (noisy_latents - sqrt_one_minus * noise_pred) / sqrt_ac.clamp(min=1e-8)


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (--device cuda) but this PyTorch build has no CUDA. "
                "In Colab: Runtime → Change runtime type → GPU, then Runtime → Restart session "
                "and run all cells. Prefer Colab's preinstalled torch; avoid overwriting it with "
                "the wrong CUDA wheel (cu124 vs cu128). If you must reinstall, pick the wheel "
                "matching your runtime from https://pytorch.org/get-started/locally/"
            )
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_checkpoint(
    save_path: Path,
    encoder: KinematicEncoder,
    unet,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    args: argparse.Namespace,
    scaler_params: dict,
    gesture_to_int: dict,
    train_mode: str,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "global_step": global_step,
        "encoder_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "scaler_params": scaler_params,
        "gesture_to_int": gesture_to_int,
        "train_mode": train_mode,
    }

    if train_mode == "lora":
        # peft stores LoRA state inside the model — extract it
        from peft import get_peft_model_state_dict

        ckpt["lora_state_dict"] = get_peft_model_state_dict(unet)
    else:
        cross_attn_state = {
            k: v
            for k, v in unet.state_dict().items()
            if "attn2.to_k" in k or "attn2.to_v" in k
        }
        ckpt["cross_attn_state_dict"] = cross_attn_state

    out = save_path / f"step_{global_step}.pt"
    torch.save(ckpt, out)
    print(f"  Checkpoint saved: {out}")


def main() -> None:
    args = _parse_args()
    use_temporal = float(args.temporal_delta_weight) > 0.0
    device = _resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"train_sd.py started | device={device} | mode={args.train_mode}", flush=True)

    # -- lazy imports so the script can parse --help without heavy deps --------
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    print(f"Loading SD components from {args.model_id} ...", flush=True)
    print("  (downloading ~4 GB on first run — this can take 10-30 min; patience)", flush=True)
    print("  loading vae ...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    print("  loading unet ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    print("  loading scheduler ...", flush=True)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    print("SD components loaded.", flush=True)

    pred_type = getattr(noise_scheduler.config, "prediction_type", None) or "epsilon"
    if use_temporal and str(pred_type).lower() != "epsilon":
        raise RuntimeError(
            f"--temporal_delta_weight requires epsilon prediction_type; "
            f"this scheduler has {pred_type!r}."
        )

    vae.to(device)
    unet.to(device)

    # Freeze VAE entirely
    vae.requires_grad_(False)
    vae.eval()

    # Freeze U-Net base, then selectively unfreeze
    unet.requires_grad_(False)

    cross_attn_dim = unet.config.cross_attention_dim  # 768 for SD1.x, 1024 for SD2.x

    if args.train_mode == "lora":
        from peft import LoraConfig, get_peft_model

        print("Attaching LoRA adapters ...", flush=True)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["to_k", "to_v", "to_q", "to_out.0"],
            lora_dropout=0.0,
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
    else:
        for name, param in unet.named_parameters():
            if "attn2.to_k" in name or "attn2.to_v" in name:
                param.requires_grad = True

    unet.train()

    # -- dataset ---------------------------------------------------------------
    print("Building dataset ... (may be slow when reading from Google Drive)", flush=True)
    if use_temporal:
        print(
            f"  temporal_delta_weight={args.temporal_delta_weight} "
            f"(pair_gap={args.temporal_pair_gap}) — frame pairs enabled",
            flush=True,
        )
    dataset = JIGSAWSDataset(
        data_root=args.data_root,
        split="train",
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
        temporal_pairs=use_temporal,
        temporal_pair_gap=args.temporal_pair_gap,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(
        f"  {len(dataset)} {'pairs' if use_temporal else 'frames'}, "
        f"{len(loader)} batches/epoch",
        flush=True,
    )

    # Serialise scaler params for checkpoint
    scaler_params = {
        "mean": dataset.scaler.mean_.tolist(),
        "scale": dataset.scaler.scale_.tolist(),
    }

    # -- kinematic encoder -----------------------------------------------------
    kin_dim = dataset._kinematics[0].shape[1] if dataset._kinematics else 76
    clip_feat = None
    if args.clip_scene_prompt:
        print("Encoding --clip_scene_prompt with CLIP text encoder (CPU) ...")
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

    # -- optimizer (encoder + unfrozen U-Net params) ---------------------------
    trainable_params = list(encoder.parameters()) + [
        p for p in unet.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Save config for reproducibility
    (save_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    # -- training loop ---------------------------------------------------------
    global_step = 0
    vae_scale = vae.config.scaling_factor

    print(f"Training on {device} | mode={args.train_mode} | epochs={args.epochs}", flush=True)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (frames, kin, gestures) in enumerate(pbar):
            frames = frames.to(device)
            kin = kin.to(device)
            gestures = gestures.to(device)

            if use_temporal:
                # Paired clips: [B, 2, 3, H, W], same DDPM t per pair.
                b, two, c_img, h_img, w_img = frames.shape
                if two != 2:
                    raise RuntimeError(f"Expected temporal batch dim 2, got {two}")
                frames_flat = frames.reshape(b * 2, c_img, h_img, w_img)
                kin_flat = kin.reshape(b * 2, -1)
                gestures_flat = gestures.reshape(b * 2)

                with torch.no_grad():
                    latents = vae.encode(frames_flat).latent_dist.sample() * vae_scale

                noise = torch.randn_like(latents)
                timesteps_pair = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (b,),
                    device=device,
                ).long()
                timesteps = timesteps_pair.repeat_interleave(2)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = encoder(kin_flat, gestures_flat)
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                loss_eps = F.mse_loss(noise_pred, noise)

                pred_x0 = _predict_original_from_epsilon(
                    noisy_latents, noise_pred, timesteps, noise_scheduler
                )
                pred_x0_paired = pred_x0.view(b, 2, *pred_x0.shape[1:])
                clean_paired = latents.view(b, 2, *latents.shape[1:])
                delta_pred = pred_x0_paired[:, 1] - pred_x0_paired[:, 0]
                delta_clean = clean_paired[:, 1] - clean_paired[:, 0]
                loss_temp = F.l1_loss(delta_pred, delta_clean)

                loss = loss_eps + float(args.temporal_delta_weight) * loss_temp
                lt_val = loss_temp.item()
            else:
                # Single-frame batch: [B, 3, H, W]
                if frames.dim() != 4:
                    raise RuntimeError(
                        "Expected single-frame tensors [B,3,H,W]; "
                        "disable --temporal_delta_weight or fix the dataset."
                    )

                # VAE encode -> latent
                with torch.no_grad():
                    latents = vae.encode(frames).latent_dist.sample() * vae_scale

                # Forward diffusion: add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = encoder(kin, gestures)

                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
                ).sample

                loss = F.mse_loss(noise_pred, noise)
                lt_val = None

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if args.save_every > 0 and global_step % args.save_every == 0:
                    _save_checkpoint(
                        save_dir,
                        encoder,
                        unet,
                        optimizer,
                        global_step,
                        args,
                        scaler_params,
                        dataset.gesture_to_int,
                        args.train_mode,
                    )

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            postfix: dict = {
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                "step": global_step,
            }
            if lt_val is not None:
                postfix["L1_Δ"] = f"{lt_val:.4f}"
            pbar.set_postfix(**postfix)

        avg = epoch_loss / max(len(loader), 1)
        print(f"  Epoch {epoch + 1} avg loss: {avg:.5f}", flush=True)

    # Final checkpoint
    _save_checkpoint(
        save_dir,
        encoder,
        unet,
        optimizer,
        global_step,
        args,
        scaler_params,
        dataset.gesture_to_int,
        args.train_mode,
    )
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
