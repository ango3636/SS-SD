"""Reusable Stable Diffusion sampler for kinematic-conditioned generation.

Wraps the VAE, U-Net (+ LoRA / cross-attn adapters), ``KinematicEncoder``,
scaler parameters and gesture mapping stored in a checkpoint produced by
``scripts/train_sd.py`` so that inference code can load SD once and then
call :meth:`SDSampler.sample` many times without paying the model-load cost
per frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from .kinematic_encoder import KinematicEncoder


def _mps_usable() -> bool:
    """Return True only if MPS is both compiled-in and actually usable.

    ``torch.backends.mps.is_available()`` can return True on systems where
    the runtime move later fails (e.g. macOS < 14 with recent PyTorch
    builds).  We probe with a 1-element tensor move so callers get a
    definitive answer.
    """
    if not getattr(torch.backends, "mps", None):
        return False
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1, device="mps")
        return True
    except Exception:
        return False


def resolve_device(requested: str | None) -> torch.device:
    """Return a torch.device, preferring CUDA > MPS > CPU when unspecified.

    When the caller specifies a device explicitly (e.g. ``--device mps``),
    we still validate it before returning: if it is unusable we raise a
    ``RuntimeError`` with a human-readable explanation instead of letting
    the failure surface later as a cryptic tensor-move stack trace.
    """
    if requested:
        req = requested.lower()
        if req.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Requested --device {requested!r} but CUDA is not "
                    "available on this machine."
                )
            return torch.device(requested)
        if req == "mps":
            if not _mps_usable():
                raise RuntimeError(
                    "Requested --device mps but the MPS backend is not "
                    "usable here.  PyTorch's MPS backend requires macOS "
                    "14.0+ (Sonoma).  Either upgrade macOS, drop the "
                    "--device flag to fall back to CPU, or run on a "
                    "CUDA machine (e.g. a Colab GPU runtime)."
                )
            return torch.device("mps")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_usable():
        return torch.device("mps")
    return torch.device("cpu")


class SDSampler:
    """Load a trained SD checkpoint once, then sample frames from
    ``(kinematics_row, gesture_int)`` pairs.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``step_*.pt`` file produced by ``train_sd.py``.
    model_id:
        HuggingFace SD model ID.  When *None*, read from the checkpoint's
        saved ``args``.
    device:
        Optional torch device override (``"cpu"`` / ``"mps"`` / ``"cuda"``).
    image_size:
        Output image resolution in pixels (default ``256``).  Must match
        the U-Net's expected latent scaling (always /8).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        image_size: int = 256,
    ) -> None:
        self.device = resolve_device(device)
        self.image_size = image_size
        self.checkpoint_path = Path(checkpoint_path)

        # -- load checkpoint --------------------------------------------------
        ckpt: Dict[str, Any] = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.ckpt = ckpt
        saved_args = ckpt.get("args", {}) or {}
        self.saved_args = saved_args
        self.train_mode = ckpt.get(
            "train_mode", saved_args.get("train_mode", "lora")
        )
        self.model_id = model_id or saved_args.get(
            "model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )

        # -- lazy imports so importing this module is cheap ------------------
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        )

        cross_attn_dim = unet.config.cross_attention_dim
        self.cross_attn_dim = cross_attn_dim
        self.in_channels = unet.config.in_channels

        # -- restore U-Net weights -------------------------------------------
        if self.train_mode == "lora":
            from peft import (
                LoraConfig,
                get_peft_model,
                set_peft_model_state_dict,
            )

            lora_rank = saved_args.get("lora_rank", 4)
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                target_modules=["to_k", "to_v", "to_q", "to_out.0"],
                lora_dropout=0.0,
            )
            unet = get_peft_model(unet, lora_config)
            set_peft_model_state_dict(unet, ckpt["lora_state_dict"])
        else:
            unet.load_state_dict(ckpt["cross_attn_state_dict"], strict=False)

        self.vae = vae.to(self.device).eval()
        self.unet = unet.to(self.device).eval()

        # -- restore KinematicEncoder ----------------------------------------
        self.gesture_to_int: Dict[str, int] = ckpt.get("gesture_to_int", {}) or {}
        self.num_gestures = max(len(self.gesture_to_int), 1)
        encoder = KinematicEncoder(
            kin_dim=76,
            num_gestures=self.num_gestures,
            seq_len=77,
            embed_dim=cross_attn_dim,
        )
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.encoder = encoder.to(self.device).eval()

        # -- scaler params ---------------------------------------------------
        scaler_params = ckpt.get("scaler_params", {}) or {}
        self.kin_mean = np.asarray(
            scaler_params.get("mean", np.zeros(76)), dtype=np.float64
        )
        self.kin_scale = np.asarray(
            scaler_params.get("scale", np.ones(76)), dtype=np.float64
        )

        # -- scheduler (shared across samples; timesteps set in sample()) ----
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.vae_scale = self.vae.config.scaling_factor

    # ------------------------------------------------------------------
    def scale_kin(self, kin_row: np.ndarray) -> np.ndarray:
        """Apply the checkpoint's StandardScaler to a raw 76-dim kin row."""
        kin = np.asarray(kin_row, dtype=np.float64).reshape(-1)
        denom = np.where(self.kin_scale == 0, 1.0, self.kin_scale)
        return (kin - self.kin_mean) / denom

    # ------------------------------------------------------------------
    def _encode_init_image(self, init_image: Any) -> torch.Tensor:
        """VAE-encode an RGB image into SD latent space.

        Accepts a PIL ``Image``, a ``np.ndarray`` (HWC, uint8 or float in
        ``[0, 1]``), or a ``torch.Tensor`` (``[3, H, W]`` in ``[-1, 1]``).
        Returns a latent tensor of shape ``[1, C, H/8, W/8]`` already
        scaled by ``vae.config.scaling_factor`` (i.e. ready to be mixed
        with scheduler noise).
        """
        if isinstance(init_image, Image.Image):
            img = init_image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(init_image, np.ndarray):
            arr = init_image
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 127.5 - 1.0
            elif arr.max() <= 1.0 + 1e-6:
                arr = arr.astype(np.float32) * 2.0 - 1.0
            else:
                arr = arr.astype(np.float32) / 127.5 - 1.0
            if arr.ndim == 3 and arr.shape[-1] == 3:
                tensor = (
                    torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                )
            else:
                raise ValueError(
                    f"init_image ndarray must be HWC RGB, got shape {arr.shape}"
                )
        elif isinstance(init_image, torch.Tensor):
            tensor = init_image
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
        else:
            raise TypeError(
                f"init_image must be PIL.Image, np.ndarray, or torch.Tensor; "
                f"got {type(init_image)!r}"
            )

        tensor = tensor.to(self.device, dtype=torch.float32).clamp(-1, 1)
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        latent = self.vae.encode(tensor).latent_dist.mean
        return latent * self.vae_scale

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        kin_row: np.ndarray,
        gesture_int: int,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        already_scaled: bool = False,
        init_image: Any = None,
        init_strength: float = 1.0,
    ) -> Image.Image:
        """Generate a single frame.

        Parameters
        ----------
        kin_row:
            Raw 76-dim kinematic vector (numpy or tensor-like).  Will be
            standardised with the checkpoint's scaler unless
            ``already_scaled=True``.
        gesture_int:
            Integer gesture class index (must be within
            ``range(self.num_gestures)``).
        num_inference_steps:
            Number of DDIM steps (default 50).
        seed:
            Optional RNG seed for reproducible noise.
        already_scaled:
            Set to True if the caller has already normalised the kinematics
            row with :meth:`scale_kin`.
        init_image:
            Optional reference RGB image used as an SDEdit / img2img
            anchor.  Accepts ``PIL.Image``, ``np.ndarray`` (HWC RGB), or
            ``torch.Tensor`` (``[3, H, W]`` in ``[-1, 1]``).  When *None*,
            standard pure-noise denoising is used.
        init_strength:
            Only used when ``init_image`` is provided.  ``1.0`` means
            full noise (equivalent to no anchoring); ``0.0`` means no
            denoising at all (returns a decode of ``init_image``).
            Typical values for temporal anchoring: ``0.5``-``0.75``.
        """
        if already_scaled:
            kin_scaled = np.asarray(kin_row, dtype=np.float64).reshape(-1)
        else:
            kin_scaled = self.scale_kin(kin_row)

        kin_tensor = (
            torch.from_numpy(kin_scaled).float().unsqueeze(0).to(self.device)
        )
        gesture_tensor = torch.tensor(
            [int(gesture_int)], dtype=torch.long, device=self.device
        )

        cond = self.encoder(kin_tensor, gesture_tensor)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        latent_h = self.image_size // 8
        latent_w = self.image_size // 8

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        if init_image is None:
            latents = torch.randn(
                1,
                self.in_channels,
                latent_h,
                latent_w,
                device=self.device,
                generator=generator,
            )
            latents = latents * self.scheduler.init_noise_sigma
            active_timesteps = self.scheduler.timesteps
        else:
            strength = float(np.clip(init_strength, 0.0, 1.0))
            init_latent = self._encode_init_image(init_image)
            init_step_count = int(num_inference_steps * strength)
            t_start = max(num_inference_steps - init_step_count, 0)
            active_timesteps = self.scheduler.timesteps[t_start:]
            if len(active_timesteps) == 0:
                # strength == 0: skip denoising entirely, just decode the init.
                latents = init_latent
            else:
                noise = torch.randn(
                    init_latent.shape,
                    device=self.device,
                    generator=generator,
                )
                latents = self.scheduler.add_noise(
                    init_latent, noise, active_timesteps[:1]
                )

        for t in active_timesteps:
            latent_input = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(
                latent_input, t, encoder_hidden_states=cond
            ).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        image = self.vae.decode(latents / self.vae_scale).sample
        image = (image.clamp(-1, 1) + 1) / 2 * 255
        image_np = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(image_np)
