"""
Audio window slicer for SnortTracker datasets.

Slices long audio files into fixed-size windows consistent with the
audio contract.  Handles overlap, labeling, and session-aware split
boundaries to prevent data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from runtime.audio_contract import (
    DTYPE,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
    zero_pad_window,
)
from dataset.manifest import LabelStatus


# ---------------------------------------------------------------------------
# SlicedWindow — the unit of training data
# ---------------------------------------------------------------------------


@dataclass
class SlicedWindow:
    """One labeled window of audio ready for feature extraction."""

    audio: np.ndarray          # float32, shape (WINDOW_SAMPLES,)
    label: LabelStatus          # positive, negative, or ignore
    session_id: str
    window_index: int           # 0-based position within the session
    start_sample: int           # sample offset from start of file
    start_seconds: float

    @property
    def is_trainable(self) -> bool:
        """True if this window should be used for training (not ignore)."""
        return self.label in (LabelStatus.POSITIVE, LabelStatus.NEGATIVE)


# ---------------------------------------------------------------------------
# Slicer
# ---------------------------------------------------------------------------


@dataclass
class Slicer:
    """Slice audio files into labeled windows.

    Parameters
    ----------
    window_samples : int
        Window size in samples (from audio contract).
    hop_samples : int
        Hop/stride between consecutive windows (from audio contract).
    """

    window_samples: int = WINDOW_SAMPLES
    hop_samples: int = HOP_SAMPLES

    def slice(
        self,
        audio: np.ndarray,
        *,
        session_id: str = "unknown",
        label: LabelStatus = LabelStatus.UNLABELED,
    ) -> list[SlicedWindow]:
        """Slice *audio* into overlapping windows.

        Parameters
        ----------
        audio : np.ndarray
            1D float32 audio array.
        session_id : str
            Session identifier for provenance tracking.
        label : LabelStatus
            Label applied to ALL windows from this audio.
        """
        audio = np.atleast_1d(np.asarray(audio, dtype=DTYPE)).ravel()
        n_samples = audio.shape[0]

        windows: list[SlicedWindow] = []
        window_index = 0
        start = 0

        while start < n_samples:
            end = min(start + self.window_samples, n_samples)
            chunk = audio[start:end]

            # Zero-pad partial final window
            if chunk.shape[0] < self.window_samples:
                chunk = zero_pad_window(chunk)

            windows.append(SlicedWindow(
                audio=chunk,
                label=label,
                session_id=session_id,
                window_index=window_index,
                start_sample=start,
                start_seconds=start / SAMPLE_RATE,
            ))

            window_index += 1
            start += self.hop_samples

        return windows

    def slice_with_annotations(
        self,
        audio: np.ndarray,
        *,
        session_id: str,
        positive_ranges: list[tuple[float, float]],
        negative_ranges: list[tuple[float, float]],
        default_label: LabelStatus = LabelStatus.IGNORE,
    ) -> list[SlicedWindow]:
        """Slice audio with time-range annotations.

        Parameters
        ----------
        audio : np.ndarray
            1D float32 audio.
        session_id : str
            Session identifier.
        positive_ranges : list[tuple[float, float]]
            List of (start_seconds, end_seconds) ranges for positive labels.
        negative_ranges : list[tuple[float, float]]
            List of (start_seconds, end_seconds) ranges for negative labels.
        default_label : LabelStatus
            Label for windows outside any annotation range.
        """
        audio = np.atleast_1d(np.asarray(audio, dtype=DTYPE)).ravel()
        n_samples = audio.shape[0]
        duration_s = n_samples / SAMPLE_RATE

        # Build a per-sample label array for fast lookup
        sample_labels = np.full(n_samples, default_label.value, dtype=object)

        for start_s, end_s in positive_ranges:
            s = max(0, int(start_s * SAMPLE_RATE))
            e = min(n_samples, int(end_s * SAMPLE_RATE))
            sample_labels[s:e] = LabelStatus.POSITIVE.value

        for start_s, end_s in negative_ranges:
            s = max(0, int(start_s * SAMPLE_RATE))
            e = min(n_samples, int(end_s * SAMPLE_RATE))
            sample_labels[s:e] = LabelStatus.NEGATIVE.value

        # If positive and negative overlap, positive wins
        for start_s, end_s in positive_ranges:
            s = max(0, int(start_s * SAMPLE_RATE))
            e = min(n_samples, int(end_s * SAMPLE_RATE))
            sample_labels[s:e] = LabelStatus.POSITIVE.value

        windows: list[SlicedWindow] = []
        window_index = 0
        start = 0

        while start < n_samples:
            end = min(start + self.window_samples, n_samples)
            chunk = audio[start:end]

            if chunk.shape[0] < self.window_samples:
                chunk = zero_pad_window(chunk)

            # Majority-vote label for this window
            window_label_strs = sample_labels[start:end]
            unique, counts = np.unique(window_label_strs, return_counts=True)
            majority_label = LabelStatus(unique[np.argmax(counts)])

            windows.append(SlicedWindow(
                audio=chunk,
                label=majority_label,
                session_id=session_id,
                window_index=window_index,
                start_sample=start,
                start_seconds=start / SAMPLE_RATE,
            ))

            window_index += 1
            start += self.hop_samples

        return windows

    def slice_file(
        self,
        path: Path,
        *,
        session_id: Optional[str] = None,
        label: LabelStatus = LabelStatus.UNLABELED,
    ) -> list[SlicedWindow]:
        """Read a WAV file and slice it into windows.

        Parameters
        ----------
        path : Path
            Path to a 16kHz mono WAV file.
        session_id : str, optional
            Defaults to the file stem.
        label : LabelStatus
            Label for all windows (use ``slice_with_annotations`` for
            per-range labels).
        """
        import wave

        sid = session_id or path.stem

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        if sample_width == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(DTYPE) / 32768.0
        elif sample_width == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(DTYPE) / 2147483648.0
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(DTYPE) / 128.0 - 1.0

        return self.slice(arr, session_id=sid, label=label)

    def count_windows(self, duration_seconds: float) -> int:
        """How many windows a recording of *duration_seconds* produces."""
        n_samples = int(duration_seconds * SAMPLE_RATE)
        if n_samples == 0:
            return 0
        # ceil(n_samples / hop_samples) — a window starts at position 0
        # and every hop_samples as long as start < n_samples
        return (n_samples + self.hop_samples - 1) // self.hop_samples
