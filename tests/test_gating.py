"""
Tests for the gating module.

Covers: RMS energy, zero-crossing rate, high-frequency energy ratio,
Gate, GateResult, BurstTracker.
"""

import math

import numpy as np
import pytest

from runtime.audio_contract import DTYPE, SAMPLE_RATE, WINDOW_SAMPLES
from runtime.config import GateConfig
from runtime.gating import (
    BurstTracker,
    Gate,
    GateResult,
    high_freq_energy_ratio,
    rms_energy,
    zero_crossing_rate,
)


# ---------------------------------------------------------------------------
# Test signal generators
# ---------------------------------------------------------------------------

def _make_sine(
    n_samples: int = WINDOW_SAMPLES,
    freq: float = 440.0,
    amp: float = 0.1,
) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(DTYPE)


def _make_white_noise(n_samples: int = WINDOW_SAMPLES, amp: float = 0.05) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(42))
    return (amp * rng.standard_normal(n_samples)).astype(DTYPE)


def _make_silence(n_samples: int = WINDOW_SAMPLES) -> np.ndarray:
    return np.zeros(n_samples, dtype=DTYPE)


# ---------------------------------------------------------------------------
# rms_energy
# ---------------------------------------------------------------------------

class TestRmsEnergy:
    def test_silence(self):
        assert rms_energy(_make_silence()) == 0.0

    def test_sine(self):
        rms = rms_energy(_make_sine(amp=0.5))
        # RMS of 0.5 * sin is 0.5 / sqrt(2) ≈ 0.3535
        assert rms == pytest.approx(0.5 / math.sqrt(2), rel=0.05)

    def test_unit_amplitude(self):
        rms = rms_energy(np.ones(WINDOW_SAMPLES, dtype=DTYPE))
        assert rms == pytest.approx(1.0)

    def test_empty(self):
        assert rms_energy(np.array([], dtype=DTYPE)) == 0.0

    def test_known_values(self):
        # [0.5, -0.5] → RMS = sqrt((0.25 + 0.25)/2) = 0.5
        rms = rms_energy(np.array([0.5, -0.5], dtype=DTYPE))
        assert rms == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# zero_crossing_rate
# ---------------------------------------------------------------------------

class TestZeroCrossingRate:
    def test_silence(self):
        # All zeros — signbit(0) is False, so no sign changes across zeros
        assert zero_crossing_rate(_make_silence()) == 0.0

    def test_sine_440Hz(self):
        # At 440 Hz, 16 kHz SR, 400 samples = 25 ms = 11 cycles
        # Each cycle has 2 zero crossings → ~22 crossings in 399 intervals
        # ZCR ≈ 22/399 ≈ 0.055
        zcr = zero_crossing_rate(_make_sine())
        assert 0.04 < zcr < 0.07

    def test_alternating(self):
        alt = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=DTYPE)
        # 4 crossings in 4 intervals = 1.0
        assert zero_crossing_rate(alt) == 1.0

    def test_no_crossings(self):
        pos = np.ones(10, dtype=DTYPE)
        assert zero_crossing_rate(pos) == 0.0

    def test_empty(self):
        assert zero_crossing_rate(np.array([], dtype=DTYPE)) == 0.0

    def test_single_sample(self):
        assert zero_crossing_rate(np.array([0.5], dtype=DTYPE)) == 0.0


# ---------------------------------------------------------------------------
# high_freq_energy_ratio
# ---------------------------------------------------------------------------

class TestHighFreqEnergyRatio:
    def test_silence(self):
        assert high_freq_energy_ratio(_make_silence()) == 0.0

    def test_low_freq_sine(self):
        # 440 Hz sine → almost all energy below 2000 Hz → ratio ≈ 0
        ratio = high_freq_energy_ratio(_make_sine(freq=440.0))
        assert ratio < 0.01

    def test_high_freq_sine(self):
        # 4000 Hz sine → all energy above 2000 Hz → ratio ≈ 1
        ratio = high_freq_energy_ratio(_make_sine(freq=4000.0), cutoff_hz=2000.0)
        assert ratio > 0.99

    def test_white_noise_has_hf_energy(self):
        # White noise has energy across the spectrum
        ratio = high_freq_energy_ratio(_make_white_noise(), cutoff_hz=2000.0)
        # Nyquist is 8000 Hz; 2000-8000 is 75% of bandwidth
        # White noise should have roughly 0.75 ratio
        assert 0.4 < ratio < 0.95

    def test_empty(self):
        assert high_freq_energy_ratio(np.array([], dtype=DTYPE)) == 0.0


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class TestGate:
    def test_silence_fails(self):
        gate = Gate()
        result = gate.evaluate(_make_silence())
        assert not result.passed
        assert "RMS too low" in result.reason

    def test_loud_sine_passes(self):
        """A loud mid-frequency burst should pass all three metrics."""
        gate = Gate()
        # Loud enough (amp 0.5 → RMS ≈ 0.35 > 0.01)
        # Mid ZCR (~0.05 in [0.02, 0.40])
        # Low HF ratio — this might fail the HF check
        # Actually, a 440 Hz sine has near-zero HF ratio, so it will fail HF
        # Let me use a higher frequency or adjust expectations
        result = gate.evaluate(_make_sine(freq=3000.0, amp=0.5))
        # High freq + loud → RMS ok, ZCR higher, HF ratio high
        assert result.passed
        assert result.rms_ok
        assert result.zcr_ok
        assert result.hf_ok

    def test_loud_burst_passes(self):
        """A loud mixed-frequency burst (simulating snort) should pass."""
        gate = Gate()
        # Mix of low + high frequencies with decent amplitude
        t = np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLE_RATE
        burst = (
            0.3 * np.sin(2.0 * math.pi * 500.0 * t)
            + 0.3 * np.sin(2.0 * math.pi * 3000.0 * t)
            + 0.05 * np.random.default_rng(42).standard_normal(WINDOW_SAMPLES)
        ).astype(DTYPE)
        result = gate.evaluate(burst)
        assert result.passed

    def test_passes_method(self):
        gate = Gate()
        assert not gate.passes(_make_silence())
        assert gate.passes(_make_sine(freq=3000.0, amp=0.5))

    def test_debug_mode_does_not_crash(self, capsys):
        gate = Gate(debug=True)
        result = gate.evaluate(_make_sine(freq=3000.0, amp=0.5))
        # Just verify it doesn't throw
        assert result.passed

    def test_custom_thresholds(self):
        cfg = GateConfig(
            rms_energy_threshold=0.5,  # very high bar
            zero_crossing_rate_min=0.1,
            zero_crossing_rate_max=0.2,
            high_freq_energy_ratio=0.8,
        )
        gate = Gate(cfg)
        # 0.1 amp sine → RMS ≈ 0.07 < 0.5 → fail
        result = gate.evaluate(_make_sine(amp=0.1))
        assert not result.passed

    def test_gate_result_fields(self):
        gate = Gate()
        result = gate.evaluate(_make_sine(freq=3000.0, amp=0.5))
        assert isinstance(result.rms, float)
        assert isinstance(result.zcr, float)
        assert isinstance(result.hf_ratio, float)
        assert isinstance(result.passed, bool)
        assert result.rms_ok
        assert result.zcr_ok
        assert result.hf_ok

    def test_repr(self):
        gate = Gate()
        result = gate.evaluate(_make_silence())
        r = repr(result)
        assert "FAIL" in r
        assert "rms=" in r
        assert "zcr=" in r


# ---------------------------------------------------------------------------
# BurstTracker
# ---------------------------------------------------------------------------

class TestBurstTracker:
    def test_initial_state(self):
        bt = BurstTracker()
        assert bt.current_duration_ms == 0.0
        assert not bt.is_plausible_snort()

    def test_single_pass_too_short(self):
        bt = BurstTracker()
        bt.update(True)
        assert bt.current_duration_ms == 25.0
        # 25 ms < 80 ms (min_event_duration_ms) → not plausible
        assert not bt.is_plausible_snort()

    def test_four_passes_plausible(self):
        bt = BurstTracker()
        for _ in range(4):
            bt.update(True)
        # 4 * 25 = 100 ms, within [80, 600] → plausible
        assert bt.current_duration_ms == 100.0
        assert bt.is_plausible_snort()

    def test_too_long_not_plausible(self):
        cfg = GateConfig(max_event_duration_ms=100)
        bt = BurstTracker(cfg)
        for _ in range(10):
            bt.update(True)
        # 10 * 25 = 250 ms > 100 → not plausible
        assert not bt.is_plausible_snort()

    def test_break_resets_counter(self):
        bt = BurstTracker()
        bt.update(True)
        bt.update(True)
        assert bt.current_duration_ms == 50.0
        # One fail resets
        bt.update(False)
        assert bt.current_duration_ms == 0.0

    def test_reset(self):
        bt = BurstTracker()
        for _ in range(5):
            bt.update(True)
        bt.reset()
        assert bt.current_duration_ms == 0.0

    def test_custom_config(self):
        cfg = GateConfig(min_event_duration_ms=50, max_event_duration_ms=200)
        bt = BurstTracker(cfg)
        bt.update(True)
        bt.update(True)
        # 50 ms at lower bound → plausible
        assert bt.current_duration_ms == 50.0
        assert bt.is_plausible_snort()
