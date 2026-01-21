"""PyTorch model definition for failure prediction."""

from __future__ import annotations

import torch
from torch import nn


class FailureClassifier(nn.Module):
    """Simple feed-forward classifier for failure prediction."""

    def __init__(self, input_dim: int = 24, hidden_dim: int = 64) -> None:
        """Initialize classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer size.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x).squeeze(-1)
