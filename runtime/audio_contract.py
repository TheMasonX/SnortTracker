"""
Audio contract for the SnortTracker pipeline.

This module defines the *single source of truth* for all audio assumptions.
Every component that touches audio — capture, gating, feature extraction,
training, inference — must conform to this contract.

Rationale
---------
If the feature pipeline changes, it is effectively a model change.  Keeping
the contract in one place and validating inputs at every boundary prevents
silent drift between training and inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants — the contract
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000          # Hz — everything uses 16 kHz
CHANNELS: int = 1                 # mono only
DTYPE = np.float32                # internal numeric type
DTYPE_STR: str = "float32"        # string form for serialization
AMPLITUDE_RANGE: tuple[float, float] = (-1.0, 1.0)

WINDOW_SIZE_MS: int = 25          # analysis window duration
HOP_SIZE_MS: int = 10             # stride between consecutive windows

# Derived — do not edit directly
WINDOW_SAMPLES: int = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)   # 400
HOP_SAMPLES: int = int(SAMPLE_RATE * HOP_SIZE_MS / 1000)         # 160


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

class ValidationStatus(Enum):
    OK = auto()
    WRONG_SAMPLE_RATE = auto()
    WRONG_CHANNELS = auto()
    WRONG_DTYPE = auto()
    CLIPPED = auto()
    EMPTY = auto()
    TOO_SHORT = auto()
    NOT_FINITE = auto()


@dataclass
class AudioInfo:
    """Metadata about an audio buffer or file."""

    sample_rate: int
    channels: int
    dtype: type
    num_samples: int
    duration_ms: float
    peak: float
    is_clipped: bool
    has_nan: bool
    has_inf: bool

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sample_rate


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------

def validate_window(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    allow_short: bool = False,
) -> ValidationStatus:
    """Validate a single audio window against the contract.

    Parameters
    ----------
    audio : np.ndarray
        1D float32 array expected.
    sample_rate : int
        Sample rate the audio was captured at.
    allow_short : bool
        If True, windows shorter than WINDOW_SAMPLES are permitted
        (e.g. end-of-stream).  The caller must handle padding.

    Returns
    -------
    ValidationStatus
    """
    if audio.size == 0:
        return ValidationStatus.EMPTY

    if sample_rate != SAMPLE_RATE:
        return ValidationStatus.WRONG_SAMPLE_RATE

    if audio.ndim != 1:
        return ValidationStatus.WRONG_CHANNELS

    if audio.dtype != DTYPE:
        return ValidationStatus.WRONG_DTYPE

    if not allow_short and audio.shape[0] < WINDOW_SAMPLES:
        return ValidationStatus.TOO_SHORT

    if not np.isfinite(audio).all():
        return ValidationStatus.NOT_FINITE

    return ValidationStatus.OK


def inspect_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> AudioInfo:
    """Return detailed metadata about an audio buffer.

    Does *not* enforce the contract — use `validate_window` for that.
    """
    original_dtype = np.asarray(audio).dtype
    audio = np.atleast_1d(np.asarray(audio, dtype=np.float64))
    flat = audio.ravel()

    return AudioInfo(
        sample_rate=sample_rate,
        channels=1 if audio.ndim == 1 else audio.shape[0],
        dtype=original_dtype,
        num_samples=flat.shape[0],
        duration_ms=(flat.shape[0] / sample_rate) * 1000.0,
        peak=float(np.abs(flat).max()),
        is_clipped=bool(np.any(np.abs(flat) >= 0.99)),
        has_nan=bool(np.isnan(flat).any()),
        has_inf=bool(np.isinf(flat).any()),
    )


# ---------------------------------------------------------------------------
# Silence / underflow handling
# ---------------------------------------------------------------------------

def is_silent(
    audio: np.ndarray,
    threshold: float = 1e-6,
) -> bool:
    """Return True if the window RMS energy is below *threshold*."""
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))
    return rms < threshold


def zero_pad_window(audio: np.ndarray, target_samples: int = WINDOW_SAMPLES) -> np.ndarray:
    """Zero-pad (or truncate) *audio* to exactly *target_samples*.

    Used for end-of-stream or underflow windows so the feature extractor
    always receives a fixed-size buffer.
    """
    audio = np.atleast_1d(np.asarray(audio, dtype=DTYPE)).ravel()
    current = audio.shape[0]

    if current == target_samples:
        return audio.copy()
    if current > target_samples:
        return audio[:target_samples].copy()

    padded = np.zeros(target_samples, dtype=DTYPE)
    padded[:current] = audio
    return padded


# ---------------------------------------------------------------------------
# File-level validation
# ---------------------------------------------------------------------------

def validate_wav_header(
    sample_rate: int,
    channels: int,
    total_samples: int,
    *,
    path: Optional[Path] = None,
) -> tuple[ValidationStatus, str]:
    """Validate a WAV file's header fields against the contract.

    Returns (status, message).
    """
    if sample_rate != SAMPLE_RATE:
        msg = (
            f"Expected sample rate {SAMPLE_RATE} Hz, got {sample_rate} Hz"
            + (f" ({path.name})" if path else "")
        )
        return ValidationStatus.WRONG_SAMPLE_RATE, msg

    if channels != CHANNELS:
        msg = (
            f"Expected {CHANNELS} channel(s), got {channels}"
            + (f" ({path.name})" if path else "")
        )
        return ValidationStatus.WRONG_CHANNELS, msg

    if total_samples == 0:
        msg = "Audio file contains 0 samples" + (f" ({path.name})" if path else "")
        return ValidationStatus.EMPTY, msg

    return ValidationStatus.OK, "ok"


# ---------------------------------------------------------------------------
# Convenience: full file check
# ---------------------------------------------------------------------------

def check_wav_file(path: Path) -> tuple[ValidationStatus, str, Optional[AudioInfo]]:
    """Validate a WAV file end-to-end against the contract.

    Reads the file, checks header + content, and returns a detailed result.

    Returns (status, message, info_or_none).
    """
    if not path.exists():
        return ValidationStatus.EMPTY, f"File not found: {path}", None

    try:
        import wave
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # WAV header check
        status, msg = validate_wav_header(sr, ch, n_frames, path=path)
        if status is not ValidationStatus.OK:
            return status, msg, None

        # Convert to float32
        sample_width = 2  # assume 16-bit PCM for now
        audio = np.frombuffer(raw, dtype=np.int16).astype(DTYPE) / 32768.0

        # If stereo, take first channel
        if ch == 2:
            audio = audio[::2]

        info = inspect_audio(audio, sample_rate=sr)

        if info.is_clipped:
            return ValidationStatus.CLIPPED, f"Audio is clipped ({path.name})", info
        if info.has_nan or info.has_inf:
            return ValidationStatus.NOT_FINITE, f"Audio contains non-finite values ({path.name})", info

        return ValidationStatus.OK, "ok", info

    except ImportError:
        # wave is stdlib — this should never happen
        return (
            ValidationStatus.EMPTY,
            "wave module unavailable",
            None,
        )
    except (EOFError, wave.Error) as exc:
        return (
            ValidationStatus.EMPTY,
            f"Malformed WAV file: {path.name} — {exc}",
            None,
        )
