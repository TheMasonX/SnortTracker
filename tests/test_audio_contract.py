"""
Tests for the audio contract module.

Covers: valid input, silence, clipping, wrong sample rate, wrong channels,
wrong dtype, underflow padding, WAV header validation.
"""

import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from runtime.audio_contract import (
    AMPLITUDE_RANGE,
    CHANNELS,
    DTYPE,
    HOP_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SIZE_MS,
    AudioInfo,
    ValidationStatus,
    check_wav_file,
    inspect_audio,
    is_silent,
    validate_wav_header,
    validate_window,
    zero_pad_window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(n_samples: int = WINDOW_SAMPLES, *, dtype=DTYPE) -> np.ndarray:
    """Return a valid dummy audio window (low-amplitude sine)."""
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    return (0.1 * np.sin(2.0 * math.pi * 440.0 * t)).astype(dtype)


def _make_wav(
    path: Path,
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> None:
    """Write a minimal 16-bit PCM WAV file for testing."""
    audio_i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


# ---------------------------------------------------------------------------
# validate_window
# ---------------------------------------------------------------------------

class TestValidateWindow:
    def test_valid_window(self):
        w = _make_window()
        assert validate_window(w) == ValidationStatus.OK

    def test_empty_array(self):
        assert validate_window(np.array([], dtype=DTYPE)) == ValidationStatus.EMPTY

    def test_wrong_sample_rate(self):
        w = _make_window()
        assert validate_window(w, sample_rate=8000) == ValidationStatus.WRONG_SAMPLE_RATE

    def test_stereo_rejected(self):
        stereo = np.zeros((2, WINDOW_SAMPLES), dtype=DTYPE)
        assert validate_window(stereo) == ValidationStatus.WRONG_CHANNELS

    def test_wrong_dtype(self):
        w = _make_window(dtype=np.float64)
        assert validate_window(w) == ValidationStatus.WRONG_DTYPE

    def test_too_short_not_allowed(self):
        w = _make_window(n_samples=100)
        assert validate_window(w, allow_short=False) == ValidationStatus.TOO_SHORT

    def test_short_allowed(self):
        w = _make_window(n_samples=100)
        assert validate_window(w, allow_short=True) == ValidationStatus.OK

    def test_nan_rejected(self):
        w = _make_window()
        w[10] = np.nan
        assert validate_window(w) == ValidationStatus.NOT_FINITE

    def test_inf_rejected(self):
        w = _make_window()
        w[10] = np.inf
        assert validate_window(w) == ValidationStatus.NOT_FINITE


# ---------------------------------------------------------------------------
# inspect_audio
# ---------------------------------------------------------------------------

class TestInspectAudio:
    def test_basic_info(self):
        w = _make_window()
        info = inspect_audio(w)
        assert info.sample_rate == SAMPLE_RATE
        assert info.channels == 1
        assert info.dtype == DTYPE
        assert info.num_samples == WINDOW_SAMPLES
        assert info.duration_ms == pytest.approx(WINDOW_SIZE_MS, rel=0.01)
        assert isinstance(info.peak, float)
        assert 0 < info.peak < 0.3  # our sine is at 0.1 amplitude

    def test_peak_detection(self):
        w = np.zeros(WINDOW_SAMPLES, dtype=DTYPE)
        w[0] = 0.95
        info = inspect_audio(w)
        assert info.peak == pytest.approx(0.95)
        assert not info.is_clipped  # 0.95 < 0.99

    def test_clipped_detection(self):
        w = np.zeros(WINDOW_SAMPLES, dtype=DTYPE)
        w[0] = 0.995
        info = inspect_audio(w)
        assert info.is_clipped

    def test_nan_flag(self):
        w = _make_window()
        w[5] = np.nan
        info = inspect_audio(w)
        assert info.has_nan

    def test_inf_flag(self):
        w = _make_window()
        w[5] = -np.inf
        info = inspect_audio(w)
        assert info.has_inf


# ---------------------------------------------------------------------------
# is_silent
# ---------------------------------------------------------------------------

class TestIsSilent:
    def test_silence(self):
        assert is_silent(np.zeros(WINDOW_SAMPLES, dtype=DTYPE))

    def test_not_silent(self):
        assert not is_silent(_make_window())

    def test_empty_is_silent(self):
        assert is_silent(np.array([], dtype=DTYPE))

    def test_custom_threshold(self):
        quiet = np.full(WINDOW_SAMPLES, 5e-7, dtype=DTYPE)
        assert is_silent(quiet, threshold=1e-6)
        assert not is_silent(quiet, threshold=1e-7)


# ---------------------------------------------------------------------------
# zero_pad_window
# ---------------------------------------------------------------------------

class TestZeroPadWindow:
    def test_exact_length_unchanged(self):
        w = _make_window()
        padded = zero_pad_window(w)
        assert padded.shape == (WINDOW_SAMPLES,)
        np.testing.assert_array_equal(padded, w)

    def test_short_padded(self):
        w = _make_window(n_samples=100)
        padded = zero_pad_window(w)
        assert padded.shape == (WINDOW_SAMPLES,)
        np.testing.assert_array_equal(padded[:100], w)
        assert (padded[100:] == 0).all()

    def test_long_truncated(self):
        w = _make_window(n_samples=600)
        truncated = zero_pad_window(w)
        assert truncated.shape == (WINDOW_SAMPLES,)
        np.testing.assert_array_equal(truncated, w[:WINDOW_SAMPLES])

    def test_empty_becomes_zeros(self):
        padded = zero_pad_window(np.array([], dtype=DTYPE))
        assert padded.shape == (WINDOW_SAMPLES,)
        assert (padded == 0).all()


# ---------------------------------------------------------------------------
# validate_wav_header
# ---------------------------------------------------------------------------

class TestValidateWavHeader:
    def test_valid_header(self):
        status, msg = validate_wav_header(SAMPLE_RATE, 1, 16000)
        assert status == ValidationStatus.OK

    def test_wrong_rate(self):
        status, msg = validate_wav_header(44100, 1, 44100)
        assert status == ValidationStatus.WRONG_SAMPLE_RATE

    def test_stereo(self):
        status, msg = validate_wav_header(SAMPLE_RATE, 2, 16000)
        assert status == ValidationStatus.WRONG_CHANNELS

    def test_empty_file(self):
        status, msg = validate_wav_header(SAMPLE_RATE, 1, 0)
        assert status == ValidationStatus.EMPTY


# ---------------------------------------------------------------------------
# check_wav_file (integration)
# ---------------------------------------------------------------------------

class TestCheckWavFile:
    def test_valid_wav(self, tmp_path: Path):
        path = tmp_path / "valid.wav"
        _make_wav(path, _make_window())
        status, msg, info = check_wav_file(path)
        assert status == ValidationStatus.OK
        assert info is not None
        assert info.sample_rate == SAMPLE_RATE

    def test_wrong_sample_rate_wav(self, tmp_path: Path):
        path = tmp_path / "wrong_sr.wav"
        _make_wav(path, _make_window(), sample_rate=44100)
        status, msg, info = check_wav_file(path)
        assert status == ValidationStatus.WRONG_SAMPLE_RATE

    def test_stereo_wav_rejected(self, tmp_path: Path):
        """Stereo WAVs are rejected — the contract requires mono."""
        path = tmp_path / "stereo.wav"
        stereo = np.column_stack([_make_window(), _make_window() * 0.5])
        i16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(i16.tobytes())
        status, msg, info = check_wav_file(path)
        assert status == ValidationStatus.WRONG_CHANNELS

    def test_missing_file(self, tmp_path: Path):
        path = tmp_path / "nope.wav"
        status, msg, info = check_wav_file(path)
        assert status == ValidationStatus.EMPTY
        assert info is None

    def test_clipped_wav(self, tmp_path: Path):
        path = tmp_path / "clipped.wav"
        w = _make_window()
        w[0] = 0.995
        _make_wav(path, w)
        status, msg, info = check_wav_file(path)
        assert status == ValidationStatus.CLIPPED
        assert info is not None
        assert info.is_clipped


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

class TestContractConstants:
    def test_derived_samples_are_sane(self):
        assert WINDOW_SAMPLES == 400   # 25 ms * 16 kHz
        assert HOP_SAMPLES == 160      # 10 ms * 16 kHz

    def test_sample_rate_is_standard(self):
        assert SAMPLE_RATE == 16000

    def test_dtype_is_float32(self):
        assert DTYPE == np.float32

    def test_amplitude_range_is_unit(self):
        assert AMPLITUDE_RANGE == (-1.0, 1.0)
