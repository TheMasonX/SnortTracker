"""
PyTorch Dataset wrapper for SnortTracker training data.

Bridges the dataset tooling (manifest, slicer, preprocessor) with
PyTorch's DataLoader.  Produces (features, label) tuples where
features are 40-dim log-Mel vectors and labels are 0/1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.manifest import Manifest, LabelStatus, DataSplit
from dataset.preprocessor import Preprocessor


class SnortDataset(Dataset):
    """PyTorch Dataset for pre-extracted snort features.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(N, n_mels)`` float32.
    labels : np.ndarray
        Shape ``(N,)`` int64 (0 or 1).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features.copy()).float()
        self.labels = torch.from_numpy(labels.copy()).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        split: Optional[str] = None,
    ) -> "SnortDataset":
        """Build a dataset from a manifest CSV.

        Parameters
        ----------
        manifest_path : Path
            Path to the manifest CSV.
        split : str, optional
            Only load sessions in this split ("train", "val", "test").
        """
        manifest = Manifest(manifest_path)
        preprocessor = Preprocessor()
        features, labels, _ = preprocessor.process_manifest(
            manifest, split=split,
        )
        return cls(features, labels)

    @classmethod
    def from_arrays(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> "SnortDataset":
        """Build a dataset directly from numpy arrays.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, n_mels)``.
        labels : np.ndarray
            Shape ``(N,)`` — 0 for negative, 1 for positive.
        """
        return cls(features, labels)


def create_dataloaders(
    manifest_path: Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders from a manifest.

    Parameters
    ----------
    manifest_path : Path
        Path to the manifest CSV.
    batch_size : int
        Batch size for training.
    num_workers : int
        DataLoader workers (0 = main process).

    Returns
    -------
    dict[str, DataLoader]
        Keys: "train", "val", "test".
    """
    loaders: dict[str, torch.utils.data.DataLoader] = {}

    for split_name in ("train", "val", "test"):
        ds = SnortDataset.from_manifest(manifest_path, split=split_name)
        if len(ds) == 0:
            loaders[split_name] = None
            continue

        shuffle = split_name == "train"
        loaders[split_name] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=(split_name == "train"),
        )

    return loaders
