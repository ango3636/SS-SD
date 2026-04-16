"""Kinematic encoder that projects JIGSAWS kinematic vectors into the
embedding space expected by Stable Diffusion's U-Net cross-attention.

The output tensor has shape ``[B, seq_len, embed_dim]`` and can be passed
directly as ``encoder_hidden_states`` to the U-Net, replacing the CLIP
text-encoder output entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KinematicEncoder(nn.Module):
    """Map a 76-dim kinematic vector + gesture label to a sequence of
    cross-attention embeddings compatible with Stable Diffusion.

    Parameters
    ----------
    kin_dim:
        Dimensionality of the input kinematic vector (default ``76``).
    num_gestures:
        Number of distinct gesture classes (default ``16``).
    seq_len:
        Token sequence length to produce.  ``77`` matches CLIP's output
        length used by SD 1.x / 2.x.
    embed_dim:
        Per-token embedding dimension.  Use ``768`` for SD 1.x and
        ``1024`` for SD 2.x.
    """

    def __init__(
        self,
        kin_dim: int = 76,
        num_gestures: int = 16,
        seq_len: int = 77,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(kin_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, seq_len * embed_dim),
        )

        self.gesture_embed = nn.Embedding(num_gestures, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

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
            ``[B, seq_len, embed_dim]`` conditioning embeddings.
        """
        x = self.mlp(kinematics)
        x = x.view(-1, self.seq_len, self.embed_dim)

        gesture_emb = self.gesture_embed(gesture_label)  # [B, embed_dim]
        x[:, 0, :] = x[:, 0, :] + gesture_emb

        return self.layer_norm(x)
