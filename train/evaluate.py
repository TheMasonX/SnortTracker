"""
Event-level evaluation metrics for SnortTracker.

Standard window-level accuracy is misleading for event detection.
An event (snort) spans multiple windows — what matters is whether
the system detects each event at least once, not whether it
classifies every individual window correctly.

Metrics
-------
- Precision: what fraction of detected events are real?
- Recall: what fraction of real events are detected?
- F1: harmonic mean of precision and recall
- False Positive Rate: detections per minute of non-snort audio
- Event-level accuracy: did we count the right number of events?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EventMetrics:
    """Event-level evaluation results."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_events_ground_truth: int = 0
    total_events_predicted: int = 0
    total_minutes: float = 0.0  # total non-snort audio duration in minutes

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positives_per_minute(self) -> float:
        return (
            self.false_positives / self.total_minutes
            if self.total_minutes > 0
            else float("inf")
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positives_per_minute": self.false_positives_per_minute,
            "total_events_gt": float(self.total_events_ground_truth),
            "total_events_pred": float(self.total_events_predicted),
        }


def compute_event_metrics(
    *,
    ground_truth_count: int,
    predicted_count: int,
    audio_duration_minutes: float,
    ground_truth_event_times: Optional[list[float]] = None,
    predicted_event_times: Optional[list[float]] = None,
    match_window_seconds: float = 0.3,
) -> EventMetrics:
    """Compute event-level metrics given ground truth and predicted counts.

    Parameters
    ----------
    ground_truth_count : int
        Number of real snort events.
    predicted_count : int
        Number of snort events the system detected.
    audio_duration_minutes : float
        Total duration of the evaluation audio.
    ground_truth_event_times : list[float], optional
        Timestamps (seconds) of ground truth events.  If provided,
        precision/recall are computed with temporal matching.
    predicted_event_times : list[float], optional
        Timestamps (seconds) of predicted events.
    match_window_seconds : float
        Max time difference for a predicted event to "match" a ground truth.

    Returns
    -------
    EventMetrics
    """
    metrics = EventMetrics(
        total_events_ground_truth=ground_truth_count,
        total_events_predicted=predicted_count,
        total_minutes=audio_duration_minutes,
    )

    if ground_truth_event_times is not None and predicted_event_times is not None:
        # Temporal matching: each ground truth event can match at most one prediction
        matched_gt = set()
        matched_pred = set()

        # Greedy matching — for each GT event, find the closest prediction
        for gt_i, gt_time in enumerate(ground_truth_event_times):
            best_dist = float("inf")
            best_j = -1
            for pred_j, pred_time in enumerate(predicted_event_times):
                if pred_j in matched_pred:
                    continue
                dist = abs(gt_time - pred_time)
                if dist < best_dist and dist <= match_window_seconds:
                    best_dist = dist
                    best_j = pred_j
            if best_j >= 0:
                matched_gt.add(gt_i)
                matched_pred.add(best_j)

        metrics.true_positives = len(matched_gt)
        metrics.false_positives = len(predicted_event_times) - len(matched_pred)
        metrics.false_negatives = len(ground_truth_event_times) - len(matched_gt)
    else:
        # Count-based only (no temporal matching)
        metrics.true_positives = min(ground_truth_count, predicted_count)
        metrics.false_positives = max(0, predicted_count - ground_truth_count)
        metrics.false_negatives = max(0, ground_truth_count - predicted_count)

    return metrics


def window_to_event_predictions(
    window_probs: np.ndarray,
    threshold: float = 0.5,
    min_event_gap_seconds: float = 0.5,
    hop_seconds: float = 0.010,
) -> list[float]:
    """Convert window-level probabilities to event timestamps.

    Merges consecutive positive windows into discrete events.
    Events separated by less than *min_event_gap_seconds* are merged.

    Parameters
    ----------
    window_probs : np.ndarray
        1D array of per-window probabilities.
    threshold : float
        Probability above which a window is considered positive.
    min_event_gap_seconds : float
        Minimum gap between distinct events.
    hop_seconds : float
        Hop duration in seconds (10 ms default).

    Returns
    -------
    list[float]
        Event timestamps in seconds (one per detected snort).
    """
    positive = window_probs >= threshold
    if not positive.any():
        return []

    # Find contiguous regions of positive windows
    event_starts: list[int] = []
    event_ends: list[int] = []

    in_event = False
    for i, is_pos in enumerate(positive):
        if is_pos and not in_event:
            event_starts.append(i)
            in_event = True
        elif not is_pos and in_event:
            event_ends.append(i - 1)
            in_event = False
    if in_event:
        event_ends.append(len(positive) - 1)

    # Convert to timestamps (center of event)
    event_times: list[float] = []
    for start_idx, end_idx in zip(event_starts, event_ends):
        center_idx = (start_idx + end_idx) / 2.0
        event_times.append(center_idx * hop_seconds)

    # Merge events closer than min_event_gap
    if len(event_times) <= 1:
        return event_times

    merged: list[float] = [event_times[0]]
    for t in event_times[1:]:
        if t - merged[-1] >= min_event_gap_seconds:
            merged.append(t)
    return merged
