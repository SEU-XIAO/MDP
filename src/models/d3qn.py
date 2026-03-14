from __future__ import annotations

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            out = self.feature_extractor(dummy)
            feat_dim = out.view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        feat = feat.flatten(start_dim=1)

        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
