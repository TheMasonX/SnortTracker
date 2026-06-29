"""
Log-Mel spectrogram feature extraction for SnortTracker.

This module is the bridge between raw audio and the classifier.
It must produce **identical** output when called from the runtime
pipeline and from the training scripts.  All parameters are read
from ``runtime.config.FeatureConfig``, and the extractor validates
itself against ``runtime.audio_contract`` at construction time.

Design
------
- Pure NumPy — no librosa or scipy dependency at inference time
- Deterministic — same audio always produces the same features
- Single-window API — ``extract(audio)`` takes one window and returns
  an ``(n_mels,)`` feature vector
- ``extract_batch(audio_segments)`` for training convenience

Mel scale
---------
The Mel scale maps frequency to perceptual pitch:

    mel(f) = 2595 * log10(1 + f / 700)

References
----------
- Davis & Mermelstein (1980), "Comparison of parametric representations
  for monosyllabic word recognition"
- McFee et al. (2015), "librosa: Audio and Music Signal Analysis in Python"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from runtime.audio_contract import DTYPE, SAMPLE_RATE, WINDOW_SAMPLES
from runtime.config import FeatureConfig


# ---------------------------------------------------------------------------
# Mel filterbank utilities
# ---------------------------------------------------------------------------


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Build a Mel filterbank matrix.

    Returns
    -------
    np.ndarray
        Shape ``(n_mels, n_fft // 2 + 1)``.  Each row is a triangular
        filter in the frequency domain.
    """
    n_bins = n_fft // 2 + 1
    bin_freqs = np.linspace(0, sample_rate / 2, n_bins)

    # Mel-spaced center frequencies
    mel_min = hz_to_mel(np.array(f_min))
    mel_max = hz_to_mel(np.array(f_max))
    mel_centers = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_centers = mel_to_hz(mel_centers)

    # Convert center freqs to bin indices
    bin_indices = np.floor((n_fft + 1) * hz_centers / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    filters = np.zeros((n_mels, n_bins), dtype=np.float64)

    for m in range(n_mels):
        left = bin_indices[m]
        center = bin_indices[m + 1]
        right = bin_indices[m + 2]

        # Rising slope
        if center > left:
            filters[m, left:center] = (
                np.arange(left, center, dtype=np.float64) - left
            ) / (center - left)

        # Falling slope
        if right > center:
            filters[m, center:right] = (
                right - np.arange(center, right, dtype=np.float64)
            ) / (right - center)

    return filters


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


@dataclass
class FeatureExtractor:
    """Log-Mel spectrogram feature extractor.

    Parameters are read from ``FeatureConfig``.  The extractor validates
    its settings against the audio contract at construction time — if
    the config is incompatible, it raises ``ValueError`` immediately.

    Parameters
    ----------
    config : FeatureConfig
        Feature extraction parameters.

    Raises
    ------
    ValueError
        If the config conflicts with the audio contract.
    """

    n_mels: int = 40
    n_fft: int = 512
    hop_length: int = 160
    f_min: float = 80.0
    f_max: float = 8000.0
    normalize: bool = True

    _mel_filters: Optional[np.ndarray] = None
    _fft_window: Optional[np.ndarray] = None

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        if config is None:
            config = FeatureConfig()

        self.n_mels = config.n_mels
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.f_min = config.f_min
        self.f_max = config.f_max
        self.normalize = config.normalize

        self._validate()

        # Precompute Mel filterbank and FFT window
        self._mel_filters = _mel_filterbank(
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            sample_rate=SAMPLE_RATE,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self._fft_window = np.hanning(self.n_fft).astype(np.float64)

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """Raise ValueError if config conflicts with audio contract."""
        if self.n_fft < WINDOW_SAMPLES:
            raise ValueError(
                f"n_fft ({self.n_fft}) must be >= window size "
                f"({WINDOW_SAMPLES}) to avoid truncation"
            )
        if self.f_min < 0:
            raise ValueError(f"f_min ({self.f_min}) must be >= 0")
        if self.f_max > SAMPLE_RATE / 2:
            raise ValueError(
                f"f_max ({self.f_max}) exceeds Nyquist ({SAMPLE_RATE / 2})"
            )
        if self.f_min >= self.f_max:
            raise ValueError(
                f"f_min ({self.f_min}) must be < f_max ({self.f_max})"
            )
        if self.n_mels < 1:
            raise ValueError(f"n_mels ({self.n_mels}) must be >= 1")

    # ------------------------------------------------------------------
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-Mel features from a single audio window.

        Parameters
        ----------
        audio : np.ndarray
            1D float32 array of length ``WINDOW_SAMPLES`` (or shorter —
            zero-padded internally).

        Returns
        -------
        np.ndarray
            1D float32 array of shape ``(n_mels,)`` — the log-Mel
            feature vector for this window.
        """
        audio = np.atleast_1d(np.asarray(audio, dtype=np.float64)).ravel()

        # Zero-pad to n_fft
        if audio.shape[0] < self.n_fft:
            padded = np.zeros(self.n_fft, dtype=np.float64)
            padded[: audio.shape[0]] = audio
        else:
            padded = audio[: self.n_fft]

        # Apply window and compute power spectrum
        windowed = padded * self._fft_window
        spec = np.abs(np.fft.rfft(windowed)) ** 2

        # Apply Mel filterbank
        mel_energies = self._mel_filters @ spec  # (n_mels,)

        # Log compression (add small epsilon to avoid log(0))
        log_mel = np.log(np.maximum(mel_energies, 1e-10))

        # Normalize to zero mean, unit variance (if enabled)
        if self.normalize:
            mean = np.mean(log_mel)
            std = np.std(log_mel)
            if std > 1e-10:
                log_mel = (log_mel - mean) / std
            else:
                log_mel = log_mel - mean  # near-silent: just center

        return log_mel.astype(DTYPE)

    # ------------------------------------------------------------------
    def extract_batch(
        self, audio_segments: np.ndarray
    ) -> np.ndarray:
        """Extract features from a batch of audio windows.

        Parameters
        ----------
        audio_segments : np.ndarray
            Shape ``(batch_size, WINDOW_SAMPLES)`` float32.

        Returns
        -------
        np.ndarray
            Shape ``(batch_size, n_mels)`` float32.
        """
        audio_segments = np.asarray(audio_segments, dtype=np.float64)
        if audio_segments.ndim == 1:
            audio_segments = audio_segments.reshape(1, -1)

        batch_size = audio_segments.shape[0]
        features = np.zeros((batch_size, self.n_mels), dtype=DTYPE)

        for i in range(batch_size):
            features[i] = self.extract(audio_segments[i])

        return features

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self.n_mels

    @property
    def mel_filterbank(self) -> np.ndarray:
        """The precomputed Mel filterbank matrix (read-only)."""
        return self._mel_filters.copy() if self._mel_filters is not None else np.array([])

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset any internal state (this extractor is stateless)."""
        pass
