# SnortTracker — Roadmap

**Last updated:** 2026-06-28  
**Current phase:** Phase 7 complete (7 of 13)  
**Next milestone:** Dataset collection (Phase 8)

---

## Project Goal

Build a playful snort counter that behaves like a step counter: continuously listens, detects **full snorts**, and increments a local count. Runs on a Raspberry Pi Zero with a USB microphone. All data stays local.

---

## Architecture

```
capture → cheap prefilter → classifier → event state machine → append-only log
```

One pipeline. Five stages. Every stage is independently testable.

---

## Phases

### ✅ Phase 1 — Repository Skeleton

- Repo structure, `.gitignore`, `README.md`
- Shared config (`runtime/config.py`) with six dataclass sections
- Package `__init__.py` files
- Smoke tests

### ✅ Phase 2 — Audio Contract

- Single source of truth for all audio assumptions (`runtime/audio_contract.py`)
- 16 kHz mono, float32, [-1.0, 1.0], 25 ms windows, 10 ms hop
- Validation, inspection, silence detection, zero-padding
- WAV file contract enforcement

### ✅ Phase 3 — Audio Capture

- `RingBuffer` — thread-safe circular buffer with stats
- `MicrophoneCapture` — live USB mic via sounddevice
- `WavFileCapture` — dry-run mode for testing without hardware
- `CaptureWindow` — audio + metadata + health flags

### ✅ Phase 4 — Cheap Gating

- RMS energy, zero-crossing rate, high-frequency energy ratio
- `Gate` — AND-combined conservative prefilter
- `GateResult` — pass/fail with per-metric diagnostics
- `BurstTracker` — event-duration estimation

---

### ✅ Phase 5 — State Machine + Logging + CLI

- [x] `runtime/state_machine.py` — idle → candidate → confirming → counted → cooldown
- [x] `runtime/logging.py` — append-only UTC-timestamped event log
- [x] `cli/main.py` — `start`, `view`, `reset`, `tail`, `status`, `purge`
- [x] `requirements.txt` — frozen dependencies
- [x] `tests/test_state_machine.py` — written BEFORE implementation (TDD)
- [x] Log rotation: max age 30 days, max size 10 MB

### ✅ Phase 6 — End-to-End Integration (v0.1)

- [x] `runtime/classifier.py` — Classifier ABC + GateHeuristicClassifier placeholder
- [x] CLI refactored to use classifier module (swap-ready for Phase 10)
- [x] `tests/test_integration.py` — full pipeline: capture → gate → classifier → SM → log (WAV fixtures)
- [x] `## Privacy` section in README — data map, purge procedure, local-first guarantees
- [x] `purge` CLI command with confirmation prompt (`-y` flag for automation)

### ✅ Phase 7 — Feature Extraction

- [x] `runtime/features.py` — `FeatureExtractor`: log-Mel spectrogram, pure NumPy
- [x] `FeatureConfig` validated against audio contract at construction
- [x] Deterministic, versioned, identical to training path
- [x] `tests/test_features.py` — 30 tests: Mel utils, filterbank, extraction, batch, edge cases

### 🔜 Phase 8 — Dataset Collection *(NEXT)*

- [ ] Record real snorts (target user, 10+ minutes)
- [ ] Record hard negatives: speech, laughter, coughs, breathing, ambient noise, silence
- [ ] Label policy: positive / negative / ignore (ambiguous → exclude)
- [ ] Gold positives: human-confirmed full snorts only
- [ ] Split by session to prevent leakage
- [ ] `dataset/` tooling: slicing, labeling, preprocessing

### Phase 9 — Model Training

- [ ] Tiny CNN architecture (conv blocks → pooling → dense → sigmoid)
- [ ] Train on desktop GPU, validate on held-out sessions
- [ ] Class weighting: target user snorts > generic positives
- [ ] Evaluate event-level performance, not window accuracy
- [ ] `train/` module: model, dataset, train, evaluate

### Phase 10 — Model Swap & Comparison

- [ ] Export trained model (ONNX)
- [ ] Quantize for Pi Zero
- [ ] Swap into runtime, compare counts against v0.1 baseline
- [ ] Event-level metrics: recall, false positives, double-counts

### Phase 11 — Pi Zero Optimization

- [ ] Profile ONNX inference latency on ARM11
- [ ] Eliminate float64 upcasts in gate hot path (compute RMS/ZCR on float32)
- [ ] Add early exit to gate (short-circuit on RMS fail)
- [ ] Remove `zero_pad_window()` copy for exact-size windows
- [ ] Replace `datetime.now()` with monotonic counter + periodic sync
- [ ] Measure GC pause frequency under sustained load

### Phase 12 — Calibration & Threshold Tuning

- [ ] `calibrate` CLI: measure ambient noise, suggest thresholds
- [ ] Tune gate sensitivity, probability threshold, confirmation windows, cooldown
- [ ] Save calibration to config, freeze settings, re-evaluate

### Phase 13 — Hardening

- [ ] Integration tests with real WAV fixtures
- [ ] Soak test: 8-hour continuous run
- [ ] Regression fixtures for every bug found
- [ ] Edge case coverage: NaN/Inf passthrough, zero-capacity buffers, 24-bit WAV, negative config values
- [ ] Pi Zero hardware smoke test (live mic, overnight)

---

## Key Design Principles

| Principle | Status |
|-----------|--------|
| Keep the runtime small — one pipeline | ✅ Enforced |
| Treat detection as a state machine | 🔜 Phase 5 |
| Treat labels as the real bottleneck | 🔜 Phase 8 |
| One shared preprocessing contract | ✅ `audio_contract.py` |
| Prefer portable inference | 🔜 Phase 10 (ONNX) |
| Measure event-level performance | 🔜 Phase 10 |
| Calibration and health checks | 🔜 Phase 12 |

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sample rate | 16 kHz mono | Sufficient for snort detection; minimizes compute |
| Window size | 25 ms (400 samples) | Standard for speech/audio event detection |
| Hop size | 10 ms (160 samples) | Balances latency vs. compute |
| Internal dtype | float32 | Smallest portable float; adequate precision |
| Amplitude range | [-1.0, 1.0] | Industry standard; matches WAV normalization |
| Model format | ONNX | Portable across runtimes; quantizable |
| Log format | Append-only plaintext CSV | Human-readable, diffable, no database needed |
| Deployment target | Raspberry Pi Zero (ARM11, 512 MB) | Conservative baseline; Zero 2 W is a bonus |

---

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `numpy` | Array operations, FFT | `pip install numpy` |
| `sounddevice` | Live microphone capture | `pip install sounddevice` |
| `pytest` | Test framework | `pip install pytest` |
| *(future)* `onnxruntime` | Model inference | `pip install onnxruntime` |
| *(future)* `scipy` | Audio processing utilities | `pip install scipy` |

---

## Current Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| `runtime/config.py` | 4 (smoke) | ✅ Passing |
| `runtime/audio_contract.py` | 35 | ✅ Passing |
| `runtime/capture.py` | 22 | ✅ Passing |
| `runtime/gating.py` | 31 | ✅ Passing |
| `runtime/state_machine.py` | 21 | ✅ Passing |
| `runtime/classifier.py` | 5 | ✅ Passing |
| `runtime/features.py` | 30 | ✅ Passing |
| Integration (pipeline) | 9 | ✅ Passing |
| **Total** | **161** | **All passing** |

---

## Privacy & Data

- All data stays local — no network code in the runtime
- Event logs are plaintext in `logs/snort_events.log`
- Log rotation: 30 days max age, 10 MB max size
- Purge: `python -m cli.main purge` wipes all logs and counts
- Raw audio in `data/raw_audio/` — purge after dataset extraction
- See `README.md## Privacy` for full data map

---

## Quick Start

```bash
git clone https://github.com/TheMasonX/SnortTracker.git
cd SnortTracker
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
python -m pytest tests/          # 92 tests
python -m cli.main start         # begin listening (Phase 5+)
python -m cli.main view          # show count
```
