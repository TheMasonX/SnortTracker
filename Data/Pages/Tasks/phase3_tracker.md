# Phase 3 — Audio Capture — Task Tracker

**Status:** ✅ Complete  
**Started:** 2026-06-28  
**Completed:** 2026-06-28  
**Goal:** Continuously collect audio with minimal overhead and no dropped-window surprises.

---

## Task List

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Install `sounddevice` dependency | ✅ Done | `sounddevice==0.5.5` with `cffi` |
| 2 | Define `CaptureWindow` dataclass with metadata | ✅ Done | audio, timestamp_utc, sample_index, window_index, health flags |
| 3 | Implement ring buffer | ✅ Done | Thread-safe, fixed-size, wrap-around, peek, stats, overrun tracking |
| 4 | Implement `MicrophoneCapture` (live) | ✅ Done | sounddevice InputStream, callback feeding ring buffer, health reporting |
| 5 | Implement `WavFileCapture` (dry-run) | ✅ Done | Reads WAV file window-by-window, context manager, contract validation |
| 6 | Implement health/stats tracking | ✅ Done | Dropped windows, overruns, buffer fill, progress % |
| 7 | Handle error conditions gracefully | ✅ Done | Device not found, file not found, wrong SR, disconnect, empty buffer |
| 8 | Write capture tests | ✅ Done | `tests/test_capture.py` — 22 tests |
| 9 | Run tests and confirm pass | ✅ Done | 61/61 pass (combined across all test files) |

---

## Exit Criteria

- [x] System can run for a meaningful period and continue returning fixed-size windows without crashing
- [x] WAV dry-run mode works for testing without hardware
- [x] Ring buffer is thread-safe and exposes health signals
- [x] Graceful handling of device not found, disconnect, empty buffer
- [x] Dropped-window counts are tracked and inspectable

---

## Files Created

| File | Purpose |
|------|---------|
| `runtime/capture.py` | `RingBuffer`, `CaptureWindow`, `MicrophoneCapture`, `WavFileCapture`, `create_capture_source()` |
| `tests/test_capture.py` | 22 tests covering ring buffer, WAV capture, metadata, health, error handling |

## Notes

- `MicrophoneCapture` uses `sounddevice.InputStream` with a callback that runs in a background PortAudio thread
- `WavFileCapture` validates the WAV header against the audio contract on `open()`
- Both capture sources expose identical `get_window() -> Optional[CaptureWindow]` API
- Ring buffer is fully thread-safe with a `threading.Lock`
- `create_capture_source()` factory selects the right source based on whether `wav_path` is given
- Combined test run: **61 passed in 0.85s** (4 smoke + 35 contract + 22 capture)
