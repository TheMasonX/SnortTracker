"""
Model training, evaluation, and export for SnortTracker.

Modules
-------
- ``model`` — Tiny CNN/MLP classifier (SnortCNN)
- ``dataset`` — PyTorch Dataset + DataLoader factories
- ``train`` — Training loop with class weighting, early stopping
- ``evaluate`` — Event-level metrics (precision, recall, F1, FP/min)
"""

from train.model import SnortCNN, create_model
from train.dataset import SnortDataset, create_dataloaders
from train.train import Trainer, TrainingConfig, TrainingMetrics
from train.evaluate import (
    EventMetrics,
    compute_event_metrics,
    window_to_event_predictions,
)

__all__ = [
    "SnortCNN",
    "create_model",
    "SnortDataset",
    "create_dataloaders",
    "Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "EventMetrics",
    "compute_event_metrics",
    "window_to_event_predictions",
]
