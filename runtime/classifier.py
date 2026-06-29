"""
Placeholder classifier for SnortTracker v0.1.

This module provides a classifier interface that uses gate metrics to
produce a heuristic probability.  In Phase 10 this will be replaced by
a real ONNX model — the ``Classifier`` base class defines the interface
that both the placeholder and the real model must satisfy.

Design
------
- ``Classifier`` — abstract interface (predict, reset, metadata)
- ``GateHeuristicClassifier`` — v0.1 placeholder: maps gate pass/fail +
  RMS energy to a probability score
- ``create_classifier()`` — factory; returns the appropriate classifier
  based on config
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from runtime.config import InferenceConfig
from runtime.gating import GateResult


# ---------------------------------------------------------------------------
# Abstract classifier interface
# ---------------------------------------------------------------------------


class Classifier(ABC):
    """Abstract interface for a snort classifier.

    All classifiers — placeholder or real model — must implement this
    interface so the pipeline can swap them without changing other code.
    """

    @abstractmethod
    def predict(self, audio: np.ndarray, gate_result: GateResult) -> float:
        """Return a probability in [0, 1] that *audio* contains a snort.

        Parameters
        ----------
        audio : np.ndarray
            1D float32 audio window (WINDOW_SAMPLES).
        gate_result : GateResult
            The prefilter result for this window, provided as context.

        Returns
        -------
        float
            Probability in [0.0, 1.0].
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g. for stateful models)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Version string for logging / reproducibility."""
        ...


# ---------------------------------------------------------------------------
# v0.1 placeholder: gate-heuristic classifier
# ---------------------------------------------------------------------------


@dataclass
class GateHeuristicClassifier(Classifier):
    """v0.1 placeholder classifier that uses gate metrics directly.

    When the gate passes, the probability is a weighted combination
    of RMS energy (loudness) and HF ratio (tonality), clamped to
    [base_prob, max_prob].  When the gate fails, probability is 0.0.

    This is NOT a real model — it exists so the full pipeline can
    be integrated and tested before training data is collected.
    """

    base_prob: float = 0.70
    max_prob: float = 0.95
    rms_scale: float = 5.0       # RMS multiplier for confidence boost
    hf_scale: float = 1.5        # HF ratio multiplier for confidence boost

    _model_name: str = "gate-heuristic-v0"
    _model_version: str = "0.1.0"

    def predict(self, audio: np.ndarray, gate_result: GateResult) -> float:
        """Produce a heuristic probability from gate metrics."""
        if not gate_result.passed:
            return 0.0

        # Gate passed — scale probability from RMS energy and HF ratio
        rms_boost = min(gate_result.rms * self.rms_scale, 0.20)
        hf_boost = min(gate_result.hf_ratio * self.hf_scale, 0.10)

        prob = self.base_prob + rms_boost + hf_boost
        return min(prob, self.max_prob)

    def reset(self) -> None:
        """No-op for stateless heuristic."""

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_classifier(cfg: Optional[InferenceConfig] = None) -> Classifier:
    """Create a classifier based on configuration.

    Currently always returns a ``GateHeuristicClassifier``.  In Phase 10
    this will detect whether an ONNX model path is configured and return
    the appropriate implementation.
    """
    if cfg is None:
        cfg = InferenceConfig()

    # Phase 10: if cfg.model_path exists and is a valid ONNX file,
    # return ONNXClassifier(cfg.model_path)

    return GateHeuristicClassifier(
        base_prob=cfg.probability_threshold,
    )
