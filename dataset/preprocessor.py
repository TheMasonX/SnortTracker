"""
Batch feature preprocessor for SnortTracker datasets.

Converts sliced audio windows into log-Mel feature arrays ready for
model training.  Wraps ``runtime.features.FeatureExtractor`` to ensure
the training preprocessing path is identical to the runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from runtime.audio_contract import DTYPE
from runtime.config import FeatureConfig
from runtime.features import FeatureExtractor
from dataset.manifest import LabelStatus, Manifest, ManifestEntry
from dataset.slicer import SlicedWindow


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


@dataclass
class Preprocessor:
    """Convert audio windows to feature arrays + label vectors.

    Wraps ``FeatureExtractor`` so training and inference use the
    exact same preprocessing pipeline.

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature extraction parameters.  Defaults match training.
    """

    _extractor: Optional[FeatureExtractor] = None
    n_mels: int = 40

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self._extractor = FeatureExtractor(config)
        self.n_mels = self._extractor.n_mels

    # ------------------------------------------------------------------
    def process_windows(
        self,
        windows: list[SlicedWindow],
        *,
        include_ignore: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[SlicedWindow]]:
        """Extract features and labels from a list of sliced windows.

        Parameters
        ----------
        windows : list[SlicedWindow]
            Windowed audio with labels.
        include_ignore : bool
            If False, windows with label ``ignore`` are excluded.

        Returns
        -------
        features : np.ndarray
            Shape ``(N, n_mels)`` float32.
        labels : np.ndarray
            Shape ``(N,)`` int (0 = negative, 1 = positive).
        kept_windows : list[SlicedWindow]
            The windows that were actually kept (for provenance).
        """
        if include_ignore:
            kept = windows
        else:
            kept = [w for w in windows if w.is_trainable]

        if not kept:
            return (
                np.zeros((0, self.n_mels), dtype=DTYPE),
                np.zeros(0, dtype=np.int64),
                [],
            )

        features = np.zeros((len(kept), self.n_mels), dtype=DTYPE)
        labels = np.zeros(len(kept), dtype=np.int64)

        for i, window in enumerate(kept):
            features[i] = self._extractor.extract(window.audio)
            labels[i] = 1 if window.label == LabelStatus.POSITIVE else 0

        return features, labels, kept

    def process_manifest(
        self,
        manifest: Manifest,
        *,
        split: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Extract features for all sessions in a manifest.

        Parameters
        ----------
        manifest : Manifest
            Dataset manifest.
        split : str, optional
            Only process sessions in this split ("train", "val", "test").

        Returns
        -------
        features : np.ndarray
            Shape ``(N, n_mels)`` float32.
        labels : np.ndarray
            Shape ``(N,)`` int.
        metadata : list[dict]
            Per-window metadata (session_id, window_index, etc.).
        """
        from dataset.manifest import DataSplit

        all_features = []
        all_labels = []
        all_meta = []

        for entry in manifest:
            if split and entry.split.value != split:
                continue
            if entry.label_status == LabelStatus.IGNORE:
                continue

            # Slice and extract
            from dataset.slicer import Slicer
            slicer = Slicer()
            windows = slicer.slice_file(
                entry.path,
                session_id=entry.session_id,
                label=entry.label_status,
            )
            features, labels, kept = self.process_windows(windows)

            if features.shape[0] > 0:
                all_features.append(features)
                all_labels.append(labels)
                for w in kept:
                    all_meta.append({
                        "session_id": w.session_id,
                        "window_index": w.window_index,
                        "start_seconds": w.start_seconds,
                        "label": w.label.value,
                    })

        if not all_features:
            return (
                np.zeros((0, self.n_mels), dtype=DTYPE),
                np.zeros(0, dtype=np.int64),
                [],
            )

        return (
            np.concatenate(all_features, axis=0),
            np.concatenate(all_labels, axis=0),
            all_meta,
        )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        return self.n_mels
