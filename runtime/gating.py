"""
Cheap gating / prefiltering for the SnortTracker runtime.

The gate exists to *reduce compute*, not to define correctness.
It uses inexpensive signal-level metrics to reject obvious
non-events before the neural classifier runs.

Metrics
-------
- RMS energy         — absolute loudness floor
- High-frequency energy ratio — distinguishes broadband noise from tonal bursts
- Zero-crossing rate — helps separate fricative/breath noise from silence

All thresholds are pulled from ``runtime.config.GateConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from runtime.audio_contract import DTYPE, SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_SIZE_MS
from runtime.config import GateConfig


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def rms_energy(audio: np.ndarray) -> float:
    """Root-mean-square energy of the audio window.

    Computed directly on float32 — precision is adequate for
    threshold comparison and avoids a wasteful upcast.
    Returns 0.0 for empty input.
    """
    audio = np.atleast_1d(np.asarray(audio, dtype=np.float32)).ravel()
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def zero_crossing_rate(audio: np.ndarray) -> float:
    """Zero-crossing rate — proportion of samples where sign changes.

    Computed directly on the input dtype.  Returns 0.0 for fewer than 2 samples.
    """
    audio = np.atleast_1d(np.asarray(audio)).ravel()
    if audio.size < 2:
        return 0.0
    crossings = np.sum(np.abs(np.diff(np.signbit(audio))))
    return float(crossings) / (audio.size - 1)


def high_freq_energy_ratio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    cutoff_hz: float = 2000.0,
) -> float:
    """Ratio of energy above *cutoff_hz* to total energy.

    Uses a simple real FFT.  Returns 0.0 for silent/empty input.
    """
    audio = np.atleast_1d(np.asarray(audio, dtype=np.float64)).ravel()
    n = audio.size
    if n < 2:
        return 0.0

    # Real FFT (rfft returns only positive frequencies)
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    total_energy = float(np.sum(spec**2))
    if total_energy < 1e-15:
        return 0.0

    hf_mask = freqs >= cutoff_hz
    hf_energy = float(np.sum(spec[hf_mask] ** 2))

    return hf_energy / total_energy


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Output of a single gate evaluation."""

    passed: bool
    rms: float
    zcr: float
    hf_ratio: float
    reason: str = ""

    # Per-metric pass/fail for diagnostics
    rms_ok: bool = True
    zcr_ok: bool = True
    hf_ok: bool = True

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"GateResult({status} | rms={self.rms:.4f} "
            f"zcr={self.zcr:.3f} hf={self.hf_ratio:.3f}"
            + (f" reason='{self.reason}'" if self.reason else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class Gate:
    """Cheap prefilter that combines RMS, ZCR, and HF-energy ratio.

    Parameters
    ----------
    config : GateConfig
        Threshold configuration.
    debug : bool
        If True, ``evaluate()`` logs detailed metrics to stderr.
    """

    def __init__(self, config: Optional[GateConfig] = None, *, debug: bool = False) -> None:
        self._cfg = config or GateConfig()
        self._debug = debug

    # -- single-window evaluation -------------------------------------------

    def evaluate(self, audio: np.ndarray) -> GateResult:
        """Run all gate metrics on *audio* and return a result.

        Metrics are evaluated in order of cheapest → most expensive.
        Evaluation short-circuits: if RMS fails, ZCR and HF ratio are skipped.
        """
        rms = rms_energy(audio)
        rms_ok = rms >= self._cfg.rms_energy_threshold

        if not rms_ok:
            result = GateResult(
                passed=False,
                rms=rms,
                zcr=0.0,
                hf_ratio=0.0,
                reason=f"RMS too low ({rms:.4f} < {self._cfg.rms_energy_threshold})",
                rms_ok=False,
                zcr_ok=True,
                hf_ok=True,
            )
            if self._debug:
                self._log(result)
            return result

        zcr = zero_crossing_rate(audio)
        zcr_ok = self._cfg.zero_crossing_rate_min <= zcr <= self._cfg.zero_crossing_rate_max

        if not zcr_ok:
            result = GateResult(
                passed=False,
                rms=rms,
                zcr=zcr,
                hf_ratio=0.0,
                reason=(
                    f"ZCR out of range ({zcr:.3f} not in "
                    f"[{self._cfg.zero_crossing_rate_min}, "
                    f"{self._cfg.zero_crossing_rate_max}])"
                ),
                rms_ok=True,
                zcr_ok=False,
                hf_ok=True,
            )
            if self._debug:
                self._log(result)
            return result

        hf = high_freq_energy_ratio(audio)
        hf_ok = hf >= self._cfg.high_freq_energy_ratio

        passed = hf_ok
        reason = ""
        if not hf_ok:
            reason = f"HF ratio too low ({hf:.3f} < {self._cfg.high_freq_energy_ratio})"

        result = GateResult(
            passed=passed,
            rms=rms,
            zcr=zcr,
            hf_ratio=hf,
            reason=reason,
            rms_ok=True,
            zcr_ok=True,
            hf_ok=hf_ok,
        )

        if self._debug:
            self._log(result)

        return result

    # -- convenience ---------------------------------------------------------

    def passes(self, audio: np.ndarray) -> bool:
        """Return True if the window passes the gate."""
        return self.evaluate(audio).passed

    # -- debug logging -------------------------------------------------------

    def _log(self, result: GateResult) -> None:
        import sys

        print(
            f"[gate] {result}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Event-duration estimator (stateful)
# ---------------------------------------------------------------------------


class BurstTracker:
    """Tracks consecutive gate passes to estimate event duration.

    This is a lightweight companion to the Gate.  It maintains a simple
    stateful counter so the pipeline can reject bursts that are too short
    or too long *before* reaching the classifier.
    """

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self._cfg = config or GateConfig()
        self._consecutive_passes = 0
        self._total_windows_observed = 0

    def update(self, gate_passed: bool, window_duration_ms: float | None = None) -> None:
        """Feed the result of one gate evaluation."""
        self._total_windows_observed += 1
        if gate_passed:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

    @property
    def current_duration_ms(self) -> float:
        """Estimated duration of the current burst in ms."""
        return self._consecutive_passes * WINDOW_SIZE_MS

    def is_plausible_snort(self) -> bool:
        """Return True if the current burst is within plausible snort duration."""
        dur = self.current_duration_ms
        if dur == 0.0:
            return False
        return self._cfg.min_event_duration_ms <= dur <= self._cfg.max_event_duration_ms

    def reset(self) -> None:
        self._consecutive_passes = 0
        self._total_windows_observed = 0
