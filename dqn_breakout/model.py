"""Q-network used by the agent."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Convolutional Q-network, as defined in the original notebook.

    Input: a stack of 4 grayscale 84x84 frames. As in the notebook, pixel
    values are fed in unscaled (0-255).
    """

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.network(torch.zeros(1, in_channels, 84, 84)).shape[1]
        self.output_layer = nn.Linear(conv_out, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.network(x))
