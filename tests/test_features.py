"""
Tests for the log-Mel spectrogram feature extractor.

Covers: construction validation, output dimensions, determinism,
Mel filterbank properties, edge cases (silence, short audio, NaN/Inf).
"""

from __future__ import annotations

import numpy as np
import pytest

from runtime.audio_contract import DTYPE, SAMPLE_RATE, WINDOW_SAMPLES
from runtime.config import FeatureConfig
from runtime.features import (
    FeatureExtractor,
    _mel_filterbank,
    hz_to_mel,
    mel_to_hz,
)


# ---------------------------------------------------------------------------
# Mel scale utilities
# ---------------------------------------------------------------------------


class TestMelScale:
    def test_hz_to_mel_zero(self):
        assert hz_to_mel(np.array(0.0)) == 0.0

    def test_mel_to_hz_zero(self):
        assert mel_to_hz(np.array(0.0)) == 0.0

    def test_roundtrip(self):
        """Hz → Mel → Hz should be identity for common frequencies."""
        test_hz = np.array([100.0, 500.0, 1000.0, 3000.0, 8000.0])
        mel = hz_to_mel(test_hz)
        recovered = mel_to_hz(mel)
        assert np.allclose(recovered, test_hz, rtol=1e-4)

    def test_monotonic(self):
        """Mel scale is monotonically increasing with Hz."""
        hz = np.linspace(0, 8000, 100)
        mel = hz_to_mel(hz)
        assert np.all(np.diff(mel) > 0)


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------


class TestMelFilterbank:
    def test_filterbank_shape(self):
        filters = _mel_filterbank(
            n_mels=40, n_fft=512, sample_rate=SAMPLE_RATE,
            f_min=80.0, f_max=8000.0,
        )
        assert filters.shape == (40, 257)  # (n_mels, n_fft//2 + 1)

    def test_filterbank_nonnegative(self):
        filters = _mel_filterbank(40, 512, SAMPLE_RATE, 80.0, 8000.0)
        assert np.all(filters >= 0)

    def test_filterbank_max_one(self):
        """Each filter should peak at 1.0."""
        filters = _mel_filterbank(40, 512, SAMPLE_RATE, 80.0, 8000.0)
        maxima = filters.max(axis=1)
        assert np.allclose(maxima, 1.0, atol=1e-10)

    def test_filterbank_sum_positive(self):
        """Frequency bins within [f_min, f_max) should have filter coverage."""
        filters = _mel_filterbank(40, 512, SAMPLE_RATE, 80.0, 8000.0)
        bin_sums = filters.sum(axis=0)
        # Bins within [f_min, f_max) should be covered
        n_bins = 257
        nyquist = SAMPLE_RATE / 2
        bin_freqs = np.linspace(0, nyquist, n_bins)
        # Exclude the exact Nyquist bin (may fall outside the last filter)
        in_range = (bin_freqs >= 80.0) & (bin_freqs < 7990.0)
        assert np.all(bin_sums[in_range] > 0), (
            "All bins in [f_min, f_max) should have filter coverage"
        )

    def test_filterbank_triangular_shape(self):
        """Each filter should be triangular (single peak, monotonic sides)."""
        filters = _mel_filterbank(20, 512, SAMPLE_RATE, 80.0, 8000.0)
        for m in range(20):
            row = filters[m]
            peak_idx = np.argmax(row)
            # Rising side
            if peak_idx > 0:
                assert np.all(np.diff(row[: peak_idx + 1]) >= -1e-12)
            # Falling side
            if peak_idx < len(row) - 1:
                assert np.all(np.diff(row[peak_idx:]) <= 1e-12)


# ---------------------------------------------------------------------------
# FeatureExtractor construction validation
# ---------------------------------------------------------------------------


class TestFeatureExtractorValidation:
    def test_default_construction(self):
        extractor = FeatureExtractor()
        assert extractor.n_mels == 40
        assert extractor.n_fft == 512
        assert extractor.output_dim == 40

    def test_from_config(self):
        cfg = FeatureConfig(n_mels=20, n_fft=1024)
        extractor = FeatureExtractor(cfg)
        assert extractor.n_mels == 20
        assert extractor.n_fft == 1024

    def test_rejects_nfft_smaller_than_window(self):
        """n_fft must be >= WINDOW_SAMPLES."""
        cfg = FeatureConfig(n_fft=256)  # < 400
        with pytest.raises(ValueError, match="n_fft"):
            FeatureExtractor(cfg)

    def test_rejects_negative_f_min(self):
        cfg = FeatureConfig(f_min=-10)
        with pytest.raises(ValueError, match="f_min"):
            FeatureExtractor(cfg)

    def test_rejects_f_max_above_nyquist(self):
        cfg = FeatureConfig(f_max=9000)  # > 8000
        with pytest.raises(ValueError, match="f_max"):
            FeatureExtractor(cfg)

    def test_rejects_f_min_above_f_max(self):
        cfg = FeatureConfig(f_min=5000, f_max=1000)
        with pytest.raises(ValueError, match="f_min"):
            FeatureExtractor(cfg)

    def test_rejects_zero_mels(self):
        cfg = FeatureConfig(n_mels=0)
        with pytest.raises(ValueError, match="n_mels"):
            FeatureExtractor(cfg)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.extractor = FeatureExtractor()

    def test_output_shape(self):
        """Single window should produce (n_mels,) feature vector."""
        audio = np.random.randn(WINDOW_SAMPLES).astype(DTYPE)
        features = self.extractor.extract(audio)
        assert features.shape == (40,)
        assert features.dtype == DTYPE

    def test_deterministic(self):
        """Same input should produce identical output."""
        rng = np.random.RandomState(42)
        audio = rng.randn(WINDOW_SAMPLES).astype(DTYPE)
        f1 = self.extractor.extract(audio)
        f2 = self.extractor.extract(audio)
        assert np.array_equal(f1, f2)

    def test_different_inputs_produce_different_features(self):
        """Different audio should produce different feature vectors."""
        rng = np.random.RandomState(42)
        a1 = rng.randn(WINDOW_SAMPLES).astype(DTYPE)
        a2 = rng.randn(WINDOW_SAMPLES).astype(DTYPE)
        assert not np.allclose(
            self.extractor.extract(a1),
            self.extractor.extract(a2),
            rtol=1e-4,
        )

    def test_silent_audio_produces_finite_output(self):
        """Silence should produce finite (not NaN, not Inf) features."""
        audio = np.zeros(WINDOW_SAMPLES, dtype=DTYPE)
        features = self.extractor.extract(audio)
        assert np.all(np.isfinite(features))
        # With normalization, silent audio should be near-zero std
        # After normalization: centered at ~0 with small std
        assert np.isfinite(features).all()

    def test_short_audio_zero_padded(self):
        """Audio shorter than n_fft should be handled via zero-padding."""
        audio = np.random.randn(200).astype(DTYPE)
        features = self.extractor.extract(audio)
        assert features.shape == (40,)
        assert np.all(np.isfinite(features))

    def test_exact_nfft_audio(self):
        """Audio exactly n_fft samples should work without issue."""
        audio = np.random.randn(512).astype(DTYPE)
        features = self.extractor.extract(audio)
        assert features.shape == (40,)
        assert np.all(np.isfinite(features))

    def test_single_sample(self):
        """Single-sample audio (edge case) should not crash."""
        audio = np.array([0.5], dtype=DTYPE)
        features = self.extractor.extract(audio)
        assert features.shape == (40,)
        assert np.all(np.isfinite(features))

    def test_loud_audio_saturates_cleanly(self):
        """Full-scale audio should not produce NaN or Inf."""
        audio = np.ones(WINDOW_SAMPLES, dtype=DTYPE) * 0.99
        features = self.extractor.extract(audio)
        assert np.all(np.isfinite(features))

    def test_features_in_reasonable_range(self):
        """Normalized features should have mean ≈ 0 and std ≈ 1."""
        rng = np.random.RandomState(42)
        features_list = []
        for _ in range(50):
            audio = rng.randn(WINDOW_SAMPLES).astype(DTYPE) * 0.3
            features_list.append(self.extractor.extract(audio))

        all_features = np.stack(features_list, axis=0)  # (50, 40)
        global_mean = np.mean(all_features)
        global_std = np.std(all_features)

        # With normalization, across many windows, mean ≈ 0, std ≈ 1
        assert abs(global_mean) < 0.5, f"Mean too far from 0: {global_mean:.3f}"
        assert 0.2 < global_std < 2.0, f"Std out of range: {global_std:.3f}"


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.extractor = FeatureExtractor()

    def test_batch_output_shape(self):
        rng = np.random.RandomState(42)
        batch = rng.randn(16, WINDOW_SAMPLES).astype(DTYPE)
        features = self.extractor.extract_batch(batch)
        assert features.shape == (16, 40)
        assert features.dtype == DTYPE

    def test_batch_equals_individual(self):
        """Batch extraction should match individual extraction."""
        rng = np.random.RandomState(42)
        batch = rng.randn(8, WINDOW_SAMPLES).astype(DTYPE)

        batch_features = self.extractor.extract_batch(batch)

        for i in range(8):
            individual = self.extractor.extract(batch[i])
            assert np.allclose(batch_features[i], individual, rtol=1e-5)

    def test_batch_single_row(self):
        """Single-row batch should work (edge case)."""
        audio = np.random.randn(WINDOW_SAMPLES).astype(DTYPE)
        features = self.extractor.extract_batch(audio)
        assert features.shape == (1, 40)

    def test_batch_empty(self):
        """Empty batch should produce empty output."""
        batch = np.zeros((0, WINDOW_SAMPLES), dtype=DTYPE)
        features = self.extractor.extract_batch(batch)
        assert features.shape == (0, 40)


# ---------------------------------------------------------------------------
# Mel filterbank access
# ---------------------------------------------------------------------------


class TestFilterbankAccess:
    def test_mel_filterbank_property(self):
        extractor = FeatureExtractor()
        fb = extractor.mel_filterbank
        assert fb.shape == (40, 257)
        assert isinstance(fb, np.ndarray)

    def test_filterbank_is_copy(self):
        """mel_filterbank property should return a copy, not a reference."""
        extractor = FeatureExtractor()
        fb1 = extractor.mel_filterbank
        fb2 = extractor.mel_filterbank
        fb1[0, 0] = 999.0
        assert fb2[0, 0] != 999.0, "Should be independent copies"


# ---------------------------------------------------------------------------
# FeatureExtractor.reset
# ---------------------------------------------------------------------------


class TestFeatureExtractorReset:
    def test_reset_is_noop(self):
        extractor = FeatureExtractor()
        extractor.reset()  # should not raise


# ---------------------------------------------------------------------------
# Config-driven feature dimensions
# ---------------------------------------------------------------------------


class TestConfigDimensions:
    def test_custom_n_mels(self):
        cfg = FeatureConfig(n_mels=64)
        extractor = FeatureExtractor(cfg)
        features = extractor.extract(
            np.random.randn(WINDOW_SAMPLES).astype(DTYPE)
        )
        assert features.shape == (64,)

    def test_custom_nfft(self):
        cfg = FeatureConfig(n_fft=1024)
        extractor = FeatureExtractor(cfg)
        features = extractor.extract(
            np.random.randn(WINDOW_SAMPLES).astype(DTYPE)
        )
        assert features.shape == (40,)
        assert extractor.mel_filterbank.shape == (40, 513)  # 1024//2 + 1
