"""
Tiny CNN/MLP classifier for snort detection.

The model consumes pre-extracted log-Mel features (40-dim vectors)
and outputs a single probability in [0, 1].  It is intentionally
tiny — <5k parameters — to fit comfortably on a Raspberry Pi Zero
after ONNX export.

Architecture
------------
Features (40) → Linear(64) → ReLU → Dropout → Linear(32) → ReLU →
Dropout → Linear(1) → Sigmoid
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class SnortCNN(nn.Module):
    """Compact fully-connected classifier for snort detection.

    Parameters
    ----------
    input_dim : int
        Feature vector dimension (default 40 = n_mels).
    hidden_dims : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, 1)`` — probabilities in [0, 1].
        """
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Path) -> None:
        """Save model state dict."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "SnortCNN":
        """Load model from state dict."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    def export_onnx(self, path: Path) -> None:
        """Export model to ONNX format.

        Parameters
        ----------
        path : Path
            Output path for the .onnx file.
        """
        self.eval()
        dummy_input = torch.randn(1, self.input_dim)
        torch.onnx.export(
            self,
            dummy_input,
            str(path),
            input_names=["features"],
            output_names=["probability"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "probability": {0: "batch_size"},
            },
            opset_version=14,
        )


def create_model(
    input_dim: int = 40,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
) -> SnortCNN:
    """Factory for the default snort classifier."""
    return SnortCNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
