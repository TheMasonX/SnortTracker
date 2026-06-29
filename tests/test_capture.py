"""
Tests for the audio capture module.

Covers: RingBuffer, WavFileCapture, CaptureWindow metadata, health stats.
MicrophoneCapture is tested via a hardware smoke test (not in CI).
"""

import math
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from runtime.audio_contract import DTYPE, SAMPLE_RATE, WINDOW_SAMPLES
from runtime.capture import (
    CaptureWindow,
    MicrophoneCapture,
    RingBuffer,
    WavFileCapture,
    create_capture_source,
    zero_pad_window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tone(n_samples: int, freq: float = 440.0, amp: float = 0.1) -> np.ndarray:
    """Generate a low-amplitude sine wave."""
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(DTYPE)


def _make_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a mono 16-bit PCM WAV file."""
    mono = np.atleast_1d(np.asarray(audio, dtype=np.float64)).ravel()
    i16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(i16.tobytes())


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------


class TestRingBuffer:
    CAP = 16000  # 1 second

    def test_initial_empty(self):
        rb = RingBuffer(self.CAP)
        assert rb.available() == 0

    def test_write_read_exact(self):
        rb = RingBuffer(self.CAP)
        data = _make_tone(WINDOW_SAMPLES)
        rb.write(data)
        assert rb.available() == WINDOW_SAMPLES
        out, partial = rb.read_window()
        assert not partial
        np.testing.assert_array_equal(out, data)
        assert rb.available() == 0

    def test_read_less_than_available(self):
        rb = RingBuffer(self.CAP)
        data = _make_tone(WINDOW_SAMPLES * 2)
        rb.write(data)
        out = rb.read(WINDOW_SAMPLES)
        assert out.shape == (WINDOW_SAMPLES,)
        np.testing.assert_array_equal(out, data[:WINDOW_SAMPLES])
        assert rb.available() == WINDOW_SAMPLES

    def test_partial_window(self):
        rb = RingBuffer(self.CAP)
        data = _make_tone(100)  # fewer than WINDOW_SAMPLES
        rb.write(data)
        out, partial = rb.read_window()
        assert partial
        assert out.shape == (100,)

    def test_wrap_around(self):
        """Write enough data to force the ring to wrap."""
        cap = WINDOW_SAMPLES * 3  # tight buffer
        rb = RingBuffer(cap)
        # Fill buffer completely
        rb.write(_make_tone(cap))
        assert rb.available() == cap
        # Read half
        rb.read(cap // 2)
        # Write more — forces wrap
        rb.write(_make_tone(WINDOW_SAMPLES))
        out = rb.read(WINDOW_SAMPLES)
        assert out.shape == (WINDOW_SAMPLES,)

    def test_overrun(self):
        """Writing more than capacity triggers overrun tracking."""
        cap = WINDOW_SAMPLES
        rb = RingBuffer(cap)
        huge = _make_tone(cap * 4)
        rb.write(huge)
        assert rb.stats["overruns"] >= 1
        assert rb.available() == cap  # only capacity stored after overrun

    def test_peek_does_not_consume(self):
        rb = RingBuffer(self.CAP)
        data = _make_tone(WINDOW_SAMPLES)
        rb.write(data)
        peeked = rb.read(WINDOW_SAMPLES, advance=False)
        np.testing.assert_array_equal(peeked, data)
        assert rb.available() == WINDOW_SAMPLES  # unchanged

    def test_reset(self):
        rb = RingBuffer(self.CAP)
        rb.write(_make_tone(500))
        rb.reset()
        assert rb.available() == 0
        assert rb.stats["total_written"] == 0

    def test_thread_safety(self):
        rb = RingBuffer(SAMPLE_RATE * 2)
        errors = []

        def writer():
            try:
                for _ in range(100):
                    rb.write(_make_tone(HOP_SAMPLES))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    rb.read(HOP_SAMPLES, advance=False)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors, f"Thread-safety errors: {errors}"


# ---------------------------------------------------------------------------
# WavFileCapture (dry-run)
# ---------------------------------------------------------------------------


class TestWavFileCapture:
    def test_reads_all_windows(self, tmp_path: Path):
        n_windows = 10
        audio = _make_tone(WINDOW_SAMPLES * n_windows)
        path = tmp_path / "test.wav"
        _make_wav(path, audio)

        cap = WavFileCapture(path)
        cap.open()
        windows = []
        while True:
            w = cap.get_window()
            if w is None:
                break
            windows.append(w)
        cap.close()

        assert len(windows) == n_windows

    def test_each_window_is_correct_size(self, tmp_path: Path):
        audio = _make_tone(WINDOW_SAMPLES * 3)
        path = tmp_path / "size.wav"
        _make_wav(path, audio)

        with WavFileCapture(path) as cap:
            w = cap.get_window()
            assert w is not None
            assert w.audio.shape == (WINDOW_SAMPLES,)
            assert w.audio.dtype == DTYPE

    def test_window_indices_are_monotonic(self, tmp_path: Path):
        audio = _make_tone(WINDOW_SAMPLES * 5)
        path = tmp_path / "mono.wav"
        _make_wav(path, audio)

        with WavFileCapture(path) as cap:
            indices = []
            while (w := cap.get_window()) is not None:
                indices.append(w.window_index)
            assert indices == list(range(5))

    def test_partial_final_window(self, tmp_path: Path):
        """If the WAV isn't a multiple of WINDOW_SAMPLES, last window is partial."""
        extra = 100
        audio = _make_tone(WINDOW_SAMPLES + extra)
        path = tmp_path / "partial.wav"
        _make_wav(path, audio)

        with WavFileCapture(path) as cap:
            w1 = cap.get_window()
            assert w1 is not None
            assert not w1.partial
            assert w1.window_index == 0

            w2 = cap.get_window()
            assert w2 is not None
            assert w2.partial
            assert w2.window_index == 1

            # end
            assert cap.get_window() is None

    def test_returns_none_at_eof(self, tmp_path: Path):
        audio = _make_tone(WINDOW_SAMPLES)
        path = tmp_path / "eof.wav"
        _make_wav(path, audio)

        with WavFileCapture(path) as cap:
            assert cap.get_window() is not None
            assert cap.get_window() is None
            assert cap.get_window() is None  # idempotent

    def test_health_stats(self, tmp_path: Path):
        audio = _make_tone(WINDOW_SAMPLES * 4)
        path = tmp_path / "health.wav"
        _make_wav(path, audio)

        with WavFileCapture(path) as cap:
            assert cap.health["total_samples"] == WINDOW_SAMPLES * 4
            assert cap.health["progress_pct"] == 0.0
            cap.get_window()
            h = cap.health
            assert h["window_index"] == 1
            assert h["progress_pct"] > 0

    def test_context_manager(self, tmp_path: Path):
        audio = _make_tone(WINDOW_SAMPLES)
        path = tmp_path / "ctx.wav"
        _make_wav(path, audio)

        with WavFileCapture(path) as cap:
            w = cap.get_window()
            assert w is not None

    def test_rejects_wrong_sample_rate(self, tmp_path: Path):
        audio = _make_tone(WINDOW_SAMPLES)
        path = tmp_path / "bad_sr.wav"
        _make_wav(path, audio, sample_rate=44100)

        with pytest.raises(ValueError, match="Invalid WAV file"):
            cap = WavFileCapture(path)
            cap.open()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            cap = WavFileCapture(Path("nonexistent.wav"))
            cap.open()


# ---------------------------------------------------------------------------
# CaptureWindow
# ---------------------------------------------------------------------------


class TestCaptureWindow:
    def test_metadata(self):
        audio = _make_tone(WINDOW_SAMPLES)
        ts = None  # placeholder
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc)
        w = CaptureWindow(
            audio=audio,
            timestamp_utc=ts,
            sample_index=0,
            window_index=7,
        )
        assert w.window_index == 7
        assert w.sample_index == 0
        assert not w.partial
        assert not w.underflow
        assert w.duration_ms == pytest.approx(25.0, rel=0.01)

    def test_partial_flag(self):
        audio = _make_tone(WINDOW_SAMPLES)
        from datetime import datetime, timezone

        w = CaptureWindow(
            audio=audio,
            timestamp_utc=datetime.now(timezone.utc),
            sample_index=0,
            window_index=0,
            partial=True,
            underflow=True,
        )
        assert w.partial
        assert w.underflow


# ---------------------------------------------------------------------------
# create_capture_source factory
# ---------------------------------------------------------------------------


class TestCreateCaptureSource:
    def test_returns_wav_capture(self, tmp_path: Path):
        audio = _make_tone(WINDOW_SAMPLES)
        path = tmp_path / "factory.wav"
        _make_wav(path, audio)

        src = create_capture_source(wav_path=path)
        assert isinstance(src, WavFileCapture)
        src.close()

    def test_returns_mic_capture(self):
        """Mic capture is constructed; we don't start it (no hardware needed)."""
        src = create_capture_source()
        assert isinstance(src, MicrophoneCapture)
        assert not src.health["running"]


# ---------------------------------------------------------------------------
# HOP_SAMPLES for the thread-safety test
# ---------------------------------------------------------------------------
from runtime.audio_contract import HOP_SAMPLES  # noqa: E402
