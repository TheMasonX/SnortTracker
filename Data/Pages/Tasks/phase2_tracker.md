# Phase 2 — Audio Contract — Task Tracker

**Status:** ✅ Complete  
**Started:** 2026-06-28  
**Completed:** 2026-06-28  
**Goal:** Make audio assumptions explicit and consistent between training and inference.

---

## Task List

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Freeze audio sampling rate (16 kHz mono) | ✅ Done | Set in `runtime/config.py` and enforced by `runtime/audio_contract.py` |
| 2 | Define window/hop sizes | ✅ Done | 25 ms window (400 samples), 10 ms hop (160 samples) |
| 3 | Define inference window vs capture window relationship | ✅ Done | Both use same contract; features derived from same buffer size |
| 4 | Create `runtime/audio_contract.py` | ✅ Done | Single module: constants, validation, inspection, padding, silence check |
| 5 | Define silence / underflow handling | ✅ Done | `is_silent()` with configurable threshold; `zero_pad_window()` for end-of-stream |
| 6 | Define clipped / malformed recording handling | ✅ Done | Clip detection at 0.99 amplitude; WAV header validation; non-finite rejection |
| 7 | Define waveform dtype and numeric range | ✅ Done | `float32`, `[-1.0, 1.0]` — enforced by `validate_window()` |
| 8 | Create audio validation utility | ✅ Done | `validate_window()`, `validate_wav_header()`, `check_wav_file()` |
| 9 | Write tests for audio contract | ✅ Done | `tests/test_audio_contract.py` — 35 tests |
| 10 | Run tests and confirm pass | ✅ Done | 39/39 pass (Phase 1 + Phase 2 combined) |

---

## Exit Criteria

- [x] Audio contract is documented in a single module
- [x] Any WAV file can be validated against the contract before entering the pipeline
- [x] Silence, underflow, and clipping are handled explicitly
- [x] Numeric range and dtype are enforced
- [x] Tests cover: valid input, silence, clipping, wrong sample rate, wrong channels, wrong dtype

---

## Notes

- `runtime/audio_contract.py` is the single source of truth for all audio assumptions
- `validate_window()` enforces the contract at runtime boundaries
- `check_wav_file()` validates WAV files end-to-end using the stdlib `wave` module
- `zero_pad_window()` ensures the feature extractor always receives fixed-size buffers
- Stereo WAVs are rejected at the contract level — mono only
- Combined test run: **39 passed in 0.39s** (4 smoke + 35 audio contract)
