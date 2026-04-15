from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.proj(x)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = (x * weights).sum(dim=1)
        return context, weights


class DetectionToPredictionModel(nn.Module):
    def __init__(self, cnn_out_dim: int = 256, lstm_hidden: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = SmallCNNEncoder(out_dim=cnn_out_dim)
        self.temporal = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = TemporalAttention(dim=lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = clip.shape
        x = clip.view(b * t, c, h, w)
        x = self.encoder(x)
        x = x.view(b, t, -1)
        x, _ = self.temporal(x)
        context, attn_weights = self.attn(x)
        logits = self.classifier(context)
        return logits, attn_weights
