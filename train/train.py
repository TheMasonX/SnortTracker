"""
Training loop for the SnortTracker classifier.

Handles class-weighted binary cross-entropy, validation,
early stopping, and model checkpointing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train.model import SnortCNN


@dataclass
class TrainingConfig:
    """Hyperparameters and training settings."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stopping_patience: int = 10
    positive_class_weight: float = 3.0  # boost snort recall
    device: str = "cpu"


@dataclass
class TrainingMetrics:
    """Per-epoch training and validation metrics."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float = float("nan")
    val_acc: float = float("nan")
    elapsed_seconds: float = 0.0


@dataclass
class Trainer:
    """Train a SnortCNN model.

    Parameters
    ----------
    model : SnortCNN
        The model to train.
    config : TrainingConfig
        Training hyperparameters.
    save_dir : Path
        Directory for checkpoints.
    """

    model: SnortCNN
    config: TrainingConfig = field(default_factory=TrainingConfig)
    save_dir: Path = Path("models")

    _history: list[TrainingMetrics] = field(default_factory=list)
    _best_val_loss: float = float("inf")
    _patience_counter: int = 0
    _criterion: Optional[nn.Module] = None
    _optimizer: Optional[optim.Optimizer] = None

    def __post_init__(self) -> None:
        self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.config.device)

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> list[TrainingMetrics]:
        """Run the full training loop.

        Returns the list of per-epoch metrics.
        """
        self._history = []
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = float("nan"), float("nan")

            if val_loader is not None and len(val_loader) > 0:
                val_loss, val_acc = self._validate(val_loader)

            elapsed = time.perf_counter() - t0

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                elapsed_seconds=elapsed,
            )
            self._history.append(metrics)

            # Early stopping on validation loss
            if val_loader is not None:
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._patience_counter = 0
                    self._save_checkpoint("best_model.pt")
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.config.early_stopping_patience:
                        break

        # Save final model
        self._save_checkpoint("final_model.pt")
        return self._history

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
            features = features.to(self.config.device)
            labels = labels.to(self.config.device).view(-1, 1)

            self._optimizer.zero_grad()
            outputs = self.model(features)
            loss = self._weighted_loss(outputs, labels)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * features.size(0)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.config.device)
                labels = labels.to(self.config.device).view(-1, 1)

                outputs = self.model(features)
                loss = self._criterion(outputs, labels)
                total_loss += loss.item() * features.size(0)
                predicted = (outputs >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    def _weighted_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """BCE loss with higher weight on positive examples."""
        pos_weight = torch.tensor(
            [self.config.positive_class_weight],
            device=self.config.device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Use logits version for numerical stability
        logits = torch.logit(outputs.clamp(1e-7, 1 - 1e-7))
        return criterion(logits, labels)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "history": self._history,
                "config": self.config,
            },
            path,
        )

    # ------------------------------------------------------------------
    @property
    def history(self) -> list[TrainingMetrics]:
        return list(self._history)
