"""
End-to-end integration tests for the SnortTracker pipeline.

Exercises the full pipeline:
    capture → gate → classifier → state machine → log

Uses WAV file fixtures (no hardware required).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from runtime.audio_contract import DTYPE, SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES
from runtime.capture import WavFileCapture, create_capture_source
from runtime.classifier import GateHeuristicClassifier, create_classifier
from runtime.config import config, GateConfig, StateMachineConfig
from runtime.gating import Gate, GateResult
from runtime.logging import EventLogger
from runtime.state_machine import StateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_silent_wav(path: Path, duration_seconds: float = 1.0) -> Path:
    """Write a silent 16kHz mono float32 WAV file."""
    import struct
    import wave

    n_samples = int(SAMPLE_RATE * duration_seconds)
    samples = np.zeros(n_samples, dtype=DTYPE)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)  # float32 = 4 bytes
        wf.setframerate(SAMPLE_RATE)
        # Write as little-endian float32
        wf.writeframes(samples.tobytes())
    return path


def _make_burst_wav(
    path: Path,
    duration_seconds: float = 2.0,
    burst_start_seconds: float = 0.5,
    burst_duration_seconds: float = 0.3,
    freq_hz: float = 3000.0,
    amplitude: float = 0.5,
) -> Path:
    """Write a WAV with a single tonal burst (simulated snort).

    The burst is a sine wave at *freq_hz* with Hann envelope.
    """
    import struct
    import wave

    n_samples = int(SAMPLE_RATE * duration_seconds)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    audio = np.zeros(n_samples, dtype=np.float64)

    # Add the burst with Hann window envelope
    burst_start = int(burst_start_seconds * SAMPLE_RATE)
    burst_len = int(burst_duration_seconds * SAMPLE_RATE)
    burst_end = min(burst_start + burst_len, n_samples)

    envelope = np.hanning(burst_end - burst_start)
    sine = np.sin(2.0 * np.pi * freq_hz * t[burst_start:burst_end])
    audio[burst_start:burst_end] = amplitude * sine * envelope

    # Also add some low-level noise to avoid total silence elsewhere
    rng = np.random.RandomState(42)
    audio += 0.002 * rng.randn(n_samples)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.astype(DTYPE).tobytes())
    return path


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_gate_heuristic_pass_produces_positive_probability(self):
        clf = GateHeuristicClassifier()
        gate_result = GateResult(
            passed=True, rms=0.15, zcr=0.10, hf_ratio=0.30,
        )
        prob = clf.predict(np.zeros(WINDOW_SAMPLES, dtype=DTYPE), gate_result)
        assert 0.0 < prob <= 1.0
        assert prob >= clf.base_prob

    def test_gate_heuristic_fail_produces_zero(self):
        clf = GateHeuristicClassifier()
        gate_result = GateResult(
            passed=False, rms=0.001, zcr=0.0, hf_ratio=0.0,
            reason="RMS too low",
        )
        prob = clf.predict(np.zeros(WINDOW_SAMPLES, dtype=DTYPE), gate_result)
        assert prob == 0.0

    def test_gate_heuristic_respects_max_prob(self):
        clf = GateHeuristicClassifier(max_prob=0.95)
        gate_result = GateResult(
            passed=True, rms=0.9, zcr=0.3, hf_ratio=0.9,
        )
        prob = clf.predict(np.zeros(WINDOW_SAMPLES, dtype=DTYPE), gate_result)
        assert prob <= 0.95

    def test_create_classifier_returns_gate_heuristic(self):
        clf = create_classifier()
        assert isinstance(clf, GateHeuristicClassifier)
        assert clf.model_name == "gate-heuristic-v0"
        assert clf.model_version == "0.1.0"

    def test_classifier_reset_is_noop(self):
        clf = GateHeuristicClassifier()
        clf.reset()  # should not raise


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end tests: capture → gate → classifier → state machine → log."""

    def test_silent_wav_produces_zero_snorts(self):
        """A silent WAV should pass no gate windows and count 0 snorts."""
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "silent.wav"
            _make_silent_wav(wav_path, duration_seconds=1.0)

            cap = create_capture_source(wav_path=wav_path)
            gate = Gate(config.gate)
            clf = create_classifier(config.inference)
            sm = StateMachine.from_config(config.state_machine)

            window_count = 0
            gate_passes = 0
            while True:
                window = cap.get_window()
                if window is None:
                    break
                window_count += 1
                result = gate.evaluate(window.audio)
                if result.passed:
                    gate_passes += 1
                prob = clf.predict(window.audio, result)
                sm.update(prob, result.passed)

            cap.close()
            assert window_count > 0, "Should process at least one window"
            assert gate_passes == 0, "Silent audio should not pass gate"
            assert sm.event_count == 0, "Silent audio should count 0 snorts"

    def test_burst_wav_counts_at_least_one_snort(self):
        """A WAV with a high-frequency burst should count at least 1 snort."""
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "burst.wav"
            _make_burst_wav(
                wav_path,
                duration_seconds=2.0,
                burst_start_seconds=0.5,
                burst_duration_seconds=0.3,
                freq_hz=3000.0,
                amplitude=0.5,
            )

            cap = create_capture_source(wav_path=wav_path)
            gate = Gate(config.gate)
            clf = create_classifier(config.inference)
            sm = StateMachine.from_config(config.state_machine)

            window_count = 0
            gate_passes = 0
            while True:
                window = cap.get_window()
                if window is None:
                    break
                window_count += 1
                result = gate.evaluate(window.audio)
                if result.passed:
                    gate_passes += 1
                prob = clf.predict(window.audio, result)
                sm.update(prob, result.passed)

            cap.close()
            assert window_count > 0
            assert gate_passes > 0, (
                "Burst WAV should pass gate on at least one window"
            )
            assert sm.event_count >= 1, (
                f"Burst WAV should count at least 1 snort, got {sm.event_count}"
            )

    def test_pipeline_with_logging(self):
        """Full pipeline writes events to the log."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_path = tmp_path / "burst.wav"
            _make_burst_wav(
                wav_path,
                duration_seconds=2.0,
                burst_start_seconds=0.5,
                burst_duration_seconds=0.3,
                freq_hz=3000.0,
                amplitude=0.5,
            )

            log_path = tmp_path / "test_events.log"
            logger = EventLogger(log_path=log_path)

            cap = create_capture_source(wav_path=wav_path)
            gate = Gate(config.gate)
            clf = create_classifier(config.inference)
            sm = StateMachine.from_config(config.state_machine)

            while True:
                window = cap.get_window()
                if window is None:
                    break
                result = gate.evaluate(window.audio)
                prob = clf.predict(window.audio, result)
                if sm.update(prob, result.passed):
                    logger.log_event(confidence=prob)

            cap.close()

            assert sm.event_count >= 1
            assert logger.count >= 1
            assert log_path.exists()

            content = log_path.read_text()
            assert "event_id=" in content
            assert "confidence=" in content

    def test_pipeline_state_reset(self):
        """State machine and classifier reset between pipeline runs."""
        clf = GateHeuristicClassifier()
        sm = StateMachine.from_config(config.state_machine)

        # Run once
        gate_result = GateResult(passed=True, rms=0.15, zcr=0.10, hf_ratio=0.30)
        clf.predict(np.zeros(WINDOW_SAMPLES, dtype=DTYPE), gate_result)
        sm.update(0.80, True)  # idle → candidate

        assert sm.state != SnortState.IDLE

        # Reset
        sm.reset()
        clf.reset()
        assert sm.state == SnortState.IDLE
        assert sm.event_count == 0

    def test_pipeline_cooldown_prevents_double_count(self):
        """After a count, cooldown prevents immediate re-counting."""
        sm = StateMachine(
            confirmation_windows=1,
            cooldown_seconds=10.0,  # long cooldown
            min_gate_agreement=1,
            probability_threshold=0.70,
        )
        clf = GateHeuristicClassifier()

        # Count a snort (single positive window with gate pass)
        gate_pass = GateResult(passed=True, rms=0.15, zcr=0.10, hf_ratio=0.30)
        prob = clf.predict(np.zeros(WINDOW_SAMPLES, dtype=DTYPE), gate_pass)
        counted = sm.update(prob, True)
        assert counted
        assert sm.event_count == 1

        # Immediate second positive should NOT count (cooldown)
        gate_pass2 = GateResult(passed=True, rms=0.20, zcr=0.12, hf_ratio=0.35)
        prob2 = clf.predict(np.zeros(WINDOW_SAMPLES, dtype=DTYPE), gate_pass2)
        counted2 = sm.update(prob2, True)
        assert not counted2, "Cooldown should prevent double count"
        assert sm.event_count == 1


# ---------------------------------------------------------------------------
# Config & pipeline consistency
# ---------------------------------------------------------------------------


class TestPipelineConsistency:
    def test_config_audio_matches_contract(self):
        """AudioConfig should derive from audio_contract."""
        assert config.audio.sample_rate == SAMPLE_RATE
        assert config.audio.window_samples == WINDOW_SAMPLES
        assert config.audio.hop_samples == HOP_SAMPLES

    def test_gate_config_has_reasonable_defaults(self):
        assert 0.0 < config.gate.rms_energy_threshold < 1.0
        assert 0.0 <= config.gate.high_freq_energy_ratio <= 1.0
        assert 0.0 <= config.gate.zero_crossing_rate_min < config.gate.zero_crossing_rate_max <= 1.0

    def test_state_machine_config_has_reasonable_defaults(self):
        assert config.state_machine.confirmation_windows >= 1
        assert config.state_machine.cooldown_seconds > 0
        assert config.state_machine.candidate_timeout_seconds > 0
        assert config.state_machine.min_gate_agreement >= 1

    def test_inference_config_has_reasonable_defaults(self):
        assert 0.0 < config.inference.probability_threshold <= 1.0


# ---------------------------------------------------------------------------
# SnortState import for state checks
# ---------------------------------------------------------------------------

from runtime.state_machine import SnortState
