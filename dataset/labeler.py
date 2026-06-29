"""
Label policy and validation for SnortTracker datasets.

The label policy defines what constitutes a positive (snort), negative
(non-snort), and ignore (ambiguous) window.  This module enforces
session-level integrity: all windows from a session share the same
data split to prevent leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dataset.manifest import DataSplit, LabelStatus, Manifest, ManifestEntry


# ---------------------------------------------------------------------------
# Label policy
# ---------------------------------------------------------------------------


@dataclass
class LabelPolicy:
    """Rules for assigning labels to audio windows.

    Parameters
    ----------
    positive_label : str
        Human-readable description of what counts as a positive.
    negative_label : str
        Human-readable description of what counts as a negative.
    ignore_criteria : list[str]
        Reasons a window should be marked ``ignore`` (excluded from training).
    require_gold_positive : bool
        If True, positives must be human-confirmed (not auto-labeled).
    """

    positive_label: str = "Full snort event (audible, distinct, non-speech)"
    negative_label: str = "Confirmed non-snort (silence, speech, cough, ambient)"
    ignore_criteria: list[str] = field(default_factory=lambda: [
        "Ambiguous — could be a partial snort or breath",
        "Overlapping with speech",
        "Edge of recording (incomplete window)",
        "Loud transient (door slam, clap, thud)",
    ])
    require_gold_positive: bool = True


# ---------------------------------------------------------------------------
# Session splitter
# ---------------------------------------------------------------------------


@dataclass
class SessionSplitter:
    """Split sessions into train/val/test while keeping sessions intact.

    All windows from a single recording session MUST stay in the same
    split.  This prevents windows from the same snort event leaking
    across train and test sets.
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total})"
            )

    def assign_splits(self, manifest: Manifest) -> None:
        """Assign train/val/test splits to all entries in the manifest.

        Only assigns to entries currently marked ``UNASSIGNED``.
        Uses deterministic shuffling (seed-based) for reproducibility.
        """
        entries = [
            e for e in manifest
            if e.split == DataSplit.UNASSIGNED
        ]
        if not entries:
            return

        # Deterministic shuffle by session_id
        rng = __import__("numpy").random.RandomState(self.seed)
        indices = rng.permutation(len(entries))

        n_train = int(len(entries) * self.train_ratio)
        n_val = int(len(entries) * self.val_ratio)
        # n_test gets the remainder

        for i, idx in enumerate(indices):
            if i < n_train:
                split = DataSplit.TRAIN
            elif i < n_train + n_val:
                split = DataSplit.VAL
            else:
                split = DataSplit.TEST
            entries[idx].split = split


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


def validate_manifest_labels(manifest: Manifest) -> list[str]:
    """Check manifest for labeling issues.

    Returns a list of warnings (empty = all good).
    """
    warnings: list[str] = []

    pos_sessions = [e.session_id for e in manifest.by_label(LabelStatus.POSITIVE)]
    neg_sessions = [e.session_id for e in manifest.by_label(LabelStatus.NEGATIVE)]

    if not pos_sessions:
        warnings.append("No positive examples in manifest")

    if not neg_sessions:
        warnings.append("No negative examples in manifest")

    # Check for sessions missing splits
    for e in manifest:
        if e.label_status != LabelStatus.IGNORE and e.split == DataSplit.UNASSIGNED:
            warnings.append(
                f"Session '{e.session_id}' is labeled '{e.label_status.value}' "
                f"but has no data split assigned"
            )

    return warnings
