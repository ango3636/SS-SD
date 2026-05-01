"""Kinematic encoder that projects JIGSAWS kinematic vectors into the
embedding space expected by Stable Diffusion's U-Net cross-attention.

The output tensor has shape ``[B, seq_len, embed_dim]`` and can be passed
directly as ``encoder_hidden_states`` to the U-Net, replacing the CLIP
text-encoder output entirely.

Optional **learnable semantic tokens** (prompt-tuning style) and a single
**frozen CLIP text embedding** (projected through a learned layer) can be
concatenated along the sequence dimension for non-kinematic scene context.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def encode_clip_scene_embedding(
    model_id: str,
    prompt: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Mean-pool SD's CLIP text encoder hidden states → ``[1, dim]``."""
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_enc = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_enc.eval()
    dev = device or torch.device("cpu")
    text_enc = text_enc.to(dev)
    with torch.no_grad():
        batch = tok(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(dev) for k, v in batch.items()}
        hidden = text_enc(**batch).last_hidden_state.float()
        pooled = hidden.mean(dim=1)
    return pooled.cpu()


class KinematicEncoder(nn.Module):
    """Map a kinematic vector + gesture label to cross-attention embeddings.

    Parameters
    ----------
    kin_dim:
        Dimensionality of the input kinematic vector (``76`` raw, or larger when
        motion derivatives are appended).
    num_gestures:
        Number of distinct gesture classes (default ``16``).
    seq_len:
        Token sequence length for the kinematic MLP head.  ``77`` matches CLIP's
        output length used by SD 1.x / 2.x.
    embed_dim:
        Per-token embedding dimension.  Use ``768`` for SD 1.x and
        ``1024`` for SD 2.x.
    num_semantic_tokens:
        Extra learnable tokens concatenated after the kinematic sequence
        (prompt-tuning / soft prompts for scene semantics).
    clip_scene_feature:
        If provided, a ``[1, embed_dim]`` tensor from the frozen CLIP text
        encoder; one projected token is appended after learnable semantics.
    """

    def __init__(
        self,
        kin_dim: int = 76,
        num_gestures: int = 16,
        seq_len: int = 77,
        embed_dim: int = 768,
        num_semantic_tokens: int = 0,
        clip_scene_feature: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_semantic_tokens = num_semantic_tokens

        self.mlp = nn.Sequential(
            nn.Linear(kin_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, seq_len * embed_dim),
        )

        self.gesture_embed = nn.Embedding(num_gestures, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if num_semantic_tokens > 0:
            self.semantic_tokens = nn.Parameter(
                torch.randn(num_semantic_tokens, embed_dim) * 0.02
            )
        else:
            self.register_parameter("semantic_tokens", None)

        self.clip_fusion: Optional[nn.Linear] = None
        if clip_scene_feature is not None:
            feat = clip_scene_feature.detach().float().clone()
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            self.register_buffer("clip_scene_buf", feat)
            self.clip_fusion = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        kinematics: torch.Tensor,
        gesture_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        kinematics:
            ``[B, kin_dim]`` normalised kinematic vectors.
        gesture_label:
            ``[B]`` integer gesture class indices.

        Returns
        -------
        torch.Tensor
            ``[B, seq_len + extras, embed_dim]`` conditioning embeddings.
        """
        x = self.mlp(kinematics)
        x = x.view(-1, self.seq_len, self.embed_dim)

        gesture_emb = self.gesture_embed(gesture_label)  # [B, embed_dim]
        x[:, 0, :] = x[:, 0, :] + gesture_emb

        x = self.layer_norm(x)

        extras: list[torch.Tensor] = []
        if self.semantic_tokens is not None:
            b = kinematics.shape[0]
            sem = self.semantic_tokens.unsqueeze(0).expand(b, -1, -1)
            extras.append(sem)
        if self.clip_fusion is not None and self.clip_scene_buf is not None:
            b = kinematics.shape[0]
            clip_in = self.clip_scene_buf.expand(b, -1)
            clip_tok = self.clip_fusion(clip_in).unsqueeze(1)
            extras.append(clip_tok)

        if not extras:
            return x
        return torch.cat([x] + extras, dim=1)
