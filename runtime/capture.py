"""
Audio capture layer for the SnortTracker runtime.

Provides two capture sources:
- ``MicrophoneCapture`` — live USB microphone via ``sounddevice``
- ``WavFileCapture`` — dry-run playback from a WAV file

Both sources feed a shared ring buffer and expose an identical
``get_window() -> CaptureWindow`` API so the rest of the pipeline
does not care where audio comes from.
"""

from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from runtime.audio_contract import (
    DTYPE,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    ValidationStatus,
    validate_window,
    zero_pad_window,
)

# ---------------------------------------------------------------------------
# Capture window — the unit of audio handed to the pipeline
# ---------------------------------------------------------------------------


@dataclass
class CaptureWindow:
    """A single fixed-size audio window plus metadata."""

    audio: np.ndarray  # 1D float32, exactly WINDOW_SAMPLES
    timestamp_utc: datetime
    sample_index: int  # monotonic sample counter at window start
    window_index: int   # monotonic window counter

    # Health flags
    is_silent: bool = False
    underflow: bool = False
    overflow: bool = False
    partial: bool = False  # True if this window was padded (end-of-stream)

    @property
    def duration_ms(self) -> float:
        return (len(self.audio) / SAMPLE_RATE) * 1000.0


# ---------------------------------------------------------------------------
# Ring buffer — thread-safe circular sample store
# ---------------------------------------------------------------------------


class RingBuffer:
    """Fixed-size thread-safe ring buffer for float32 audio samples."""

    def __init__(self, capacity_samples: int) -> None:
        self._buf = np.zeros(capacity_samples, dtype=DTYPE)
        self._capacity = capacity_samples
        self._write_pos = 0
        self._total_written = 0
        self._total_read = 0
        self._overruns = 0
        self._lock = threading.Lock()

    # -- write side ----------------------------------------------------------

    def write(self, samples: np.ndarray) -> int:
        """Write samples into the buffer.  Returns number written (may be
        less than len(samples) if buffer overruns)."""
        samples = np.atleast_1d(np.asarray(samples, dtype=DTYPE)).ravel()
        n = samples.shape[0]
        if n == 0:
            return 0

        with self._lock:
            capacity = self._capacity
            if n > capacity:
                # Writer is too far behind — keep only the latest *capacity* samples
                self._buf[:] = samples[n - capacity :]
                self._write_pos = 0
                self._total_written += capacity
                self._overruns += 1
                return capacity

            space = capacity - self._write_pos
            if n <= space:
                self._buf[self._write_pos : self._write_pos + n] = samples
                self._write_pos = (self._write_pos + n) % capacity
            else:
                # Wrap
                self._buf[self._write_pos :] = samples[:space]
                remainder = n - space
                self._buf[:remainder] = samples[space:]
                self._write_pos = remainder

            self._total_written += n
            return n

    # -- read side -----------------------------------------------------------

    def available(self) -> int:
        """Number of unread samples in the buffer."""
        with self._lock:
            return max(0, self._total_written - self._total_read)

    def read(self, n_samples: int, *, advance: bool = True) -> np.ndarray:
        """Read up to *n_samples* from the buffer.

        Parameters
        ----------
        n_samples : int
            Number of samples to read.
        advance : bool
            If True, the read pointer advances (consuming the samples).
            If False, this is a non-destructive peek.

        Returns
        -------
        np.ndarray
            1D float32 array of length min(n_samples, available).
        """
        with self._lock:
            avail = max(0, self._total_written - self._total_read)
            n = min(n_samples, avail)
            if n == 0:
                return np.zeros(0, dtype=DTYPE)

            out = np.zeros(n, dtype=DTYPE)
            read_pos = self._total_read % self._capacity
            space = self._capacity - read_pos

            if n <= space:
                out[:] = self._buf[read_pos : read_pos + n]
            else:
                out[:space] = self._buf[read_pos:]
                out[space:] = self._buf[: n - space]

            if advance:
                self._total_read += n

            return out

    def read_window(self, *, advance: bool = True) -> tuple[np.ndarray, bool]:
        """Read exactly WINDOW_SAMPLES.  Returns (audio, partial) where
        partial=True means fewer than WINDOW_SAMPLES were available."""
        chunk = self.read(WINDOW_SAMPLES, advance=advance)
        partial = chunk.shape[0] < WINDOW_SAMPLES
        return chunk, partial

    # -- health --------------------------------------------------------------

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "capacity_samples": self._capacity,
                "total_written": self._total_written,
                "total_read": self._total_read,
                "available": max(0, self._total_written - self._total_read),
                "overruns": self._overruns,
            }

    def reset(self) -> None:
        with self._lock:
            self._buf.fill(0.0)
            self._write_pos = 0
            self._total_written = 0
            self._total_read = 0
            self._overruns = 0


# ---------------------------------------------------------------------------
# Microphone capture (live)
# ---------------------------------------------------------------------------


class MicrophoneCapture:
    """Live audio capture from a USB microphone via ``sounddevice``.

    Runs a background thread that reads from the device and feeds a
    ring buffer.  Call ``get_window()`` from the main thread to pull
    fixed-size windows for the pipeline.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        buffer_capacity_seconds: float = 5.0,
    ) -> None:
        self._ring = RingBuffer(int(SAMPLE_RATE * buffer_capacity_seconds))
        self._device = device
        self._running = False
        self._stream: Optional["sounddevice.InputStream"] = None  # type: ignore[name-defined]
        self._thread: Optional[threading.Thread] = None
        self._window_index = 0
        self._sample_index = 0
        self._dropped_windows = 0
        self._start_time: Optional[datetime] = None

        # Lazy import to keep sounddevice optional
        import sounddevice as _sd

        self._sd = _sd

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Open the microphone and begin streaming into the ring buffer."""
        if self._running:
            return

        # Log available devices for debugging
        try:
            devices = self._sd.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]
        except Exception:
            input_devices = []

        if not input_devices:
            raise RuntimeError(
                "No audio input devices found. Connect a USB microphone "
                "and try again, or use WavFileCapture for dry-run mode."
            )

        device = self._device
        if device is None:
            device = input_devices[0].get("index", 0)

        self._start_time = datetime.now(timezone.utc)
        self._running = True

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                # Under/overflow or other PortAudio status flag
                pass
            self._ring.write(indata[:, 0].copy() if indata.ndim > 1 else indata.ravel())

        try:
            self._stream = self._sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device,
                callback=_callback,
                blocksize=HOP_SAMPLES,
            )
            self._stream.start()
        except self._sd.PortAudioError as exc:
            self._running = False
            raise RuntimeError(
                f"Failed to open audio device: {exc}\n"
                f"Available devices:\n{self._sd.query_devices()}"
            ) from exc

    def stop(self) -> None:
        """Stop streaming and close the device."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # -- window interface ----------------------------------------------------

    def get_window(self) -> Optional[CaptureWindow]:
        """Return the next fixed-size window of audio, or None if not
        enough data is available yet."""
        chunk, partial = self._ring.read_window()

        if chunk.shape[0] == 0:
            return None

        padded = zero_pad_window(chunk)

        window = CaptureWindow(
            audio=padded,
            timestamp_utc=datetime.now(timezone.utc),
            sample_index=self._sample_index,
            window_index=self._window_index,
            underflow=partial,
            partial=partial,
        )

        self._sample_index += WINDOW_SAMPLES
        self._window_index += 1

        if partial:
            self._dropped_windows += 1

        return window

    # -- health --------------------------------------------------------------

    @property
    def health(self) -> dict:
        return {
            "running": self._running,
            "window_index": self._window_index,
            "dropped_windows": self._dropped_windows,
            "ring_buffer": self._ring.stats,
            "start_time_utc": self._start_time.isoformat() if self._start_time else None,
        }


# ---------------------------------------------------------------------------
# WAV file capture (dry-run)
# ---------------------------------------------------------------------------


class WavFileCapture:
    """Reads audio from a WAV file as if it were a live stream.

    Each call to ``get_window()`` advances the file position by
    WINDOW_SAMPLES.  Returns ``None`` at end-of-file.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._file: Optional[wave.Wave_read] = None
        self._position = 0
        self._total_samples = 0
        self._window_index = 0
        self._dropped_windows = 0
        self._finished = False

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        """Open the WAV file and validate against the audio contract."""
        if not self._path.exists():
            raise FileNotFoundError(f"WAV file not found: {self._path}")

        self._file = wave.open(str(self._path), "rb")

        sr = self._file.getframerate()
        ch = self._file.getnchannels()
        n_frames = self._file.getnframes()

        status, msg = ValidationStatus.OK, "ok"
        if sr != SAMPLE_RATE:
            status = ValidationStatus.WRONG_SAMPLE_RATE
            msg = f"Expected {SAMPLE_RATE} Hz, got {sr} Hz"
        elif ch not in (1, 2):
            status = ValidationStatus.WRONG_CHANNELS
            msg = f"Expected 1 or 2 channels, got {ch}"

        if status is not ValidationStatus.OK:
            self._file.close()
            self._file = None
            raise ValueError(f"Invalid WAV file ({self._path.name}): {msg}")

        self._total_samples = n_frames
        self._position = 0
        self._finished = False

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # -- window interface ----------------------------------------------------

    def get_window(self) -> Optional[CaptureWindow]:
        """Return the next window, or None at end-of-file."""
        if self._finished or self._file is None:
            return None

        remaining = self._total_samples - self._position
        n_to_read = min(WINDOW_SAMPLES, remaining)

        if n_to_read == 0:
            self._finished = True
            return None

        raw = self._file.readframes(n_to_read)
        self._position += n_to_read

        # Convert bytes → float32 [-1, 1]
        sample_width = self._file.getsampwidth()
        ch = self._file.getnchannels()

        if sample_width == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(DTYPE) / 32768.0
        elif sample_width == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(DTYPE) / 2147483648.0
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(DTYPE) / 128.0 - 1.0

        # If stereo, take only the first channel
        if ch == 2:
            arr = arr[::2]

        # Ensure exactly WINDOW_SAMPLES
        partial = arr.shape[0] < WINDOW_SAMPLES
        padded = zero_pad_window(arr)

        window = CaptureWindow(
            audio=padded,
            timestamp_utc=datetime.now(timezone.utc),
            sample_index=self._position - n_to_read,
            window_index=self._window_index,
            partial=partial,
            underflow=partial,
        )

        self._window_index += 1

        if partial:
            self._dropped_windows += 1
            self._finished = True

        return window

    # -- health --------------------------------------------------------------

    @property
    def health(self) -> dict:
        pct = (self._position / self._total_samples * 100) if self._total_samples > 0 else 0.0
        return {
            "source": str(self._path),
            "total_samples": self._total_samples,
            "position": self._position,
            "progress_pct": round(pct, 1),
            "window_index": self._window_index,
            "dropped_windows": self._dropped_windows,
            "finished": self._finished,
        }

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "WavFileCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Capture source factory
# ---------------------------------------------------------------------------


def create_capture_source(
    *,
    wav_path: Optional[Path] = None,
    device: Optional[int] = None,
    buffer_capacity_seconds: float = 5.0,
) -> MicrophoneCapture | WavFileCapture:
    """Create the appropriate capture source.

    If *wav_path* is given, returns ``WavFileCapture`` (dry-run).
    Otherwise returns ``MicrophoneCapture`` (live).
    """
    if wav_path is not None:
        cap = WavFileCapture(wav_path)
        cap.open()
        return cap
    return MicrophoneCapture(device=device, buffer_capacity_seconds=buffer_capacity_seconds)


# ---------------------------------------------------------------------------
# HOP_SAMPLES import (local alias for callback blocksize)
# ---------------------------------------------------------------------------
from runtime.audio_contract import HOP_SAMPLES  # noqa: E402
