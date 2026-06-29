# SnortTracker — Handoff Document

**Date:** 2026-06-28 (updated after Phases 6–7)  
**Project phase:** Phases 1–7 complete (7 of 13)  
**Test suite:** 161/161 passing  
**Working prototype:** ✅ CLI dry-run mode counts snort-like bursts from WAV files; features pipeline ready for model training  

---

## 1. Project Overview

SnortTracker is a playful snort counter — a "step counter for snorts." It continuously listens through a microphone, detects full snort events, and increments a local count. The MVP runs on a Raspberry Pi Zero with a USB microphone. All data stays local.

**Core pipeline:**
```
capture → cheap prefilter → classifier → event state machine → append-only log
```

---

## 2. What's Built (Phases 1–7)

### Phase 1 — Repository Skeleton
- Full directory structure: `runtime/`, `dataset/`, `train/`, `cli/`, `models/`, `logs/`, `tests/`
- `.gitignore`, `README.md`, `ROADMAP.md`
- Shared config (`runtime/config.py`) — six dataclass sections with safe defaults

### Phase 2 — Audio Contract
- `runtime/audio_contract.py` — single source of truth for all audio constants
- 16 kHz mono, float32, [-1.0, 1.0], 25 ms windows (400 samples), 10 ms hop (160 samples)
- Validation, inspection, silence detection, zero-padding, WAV file contract enforcement
- 35 tests

### Phase 3 — Audio Capture
- `Runtime/capture.py` — `RingBuffer` (thread-safe), `MicrophoneCapture` (live via sounddevice), `WavFileCapture` (dry-run from WAV files)
- `CaptureWindow` dataclass with audio + metadata + health flags
- `create_capture_source()` factory
- 22 tests

### Phase 4 — Cheap Gating
- `runtime/gating.py` — `rms_energy()`, `zero_crossing_rate()`, `high_freq_energy_ratio()`
- `Gate` class with AND-combined conservative filter, early exit optimization, debug mode
- `GateResult` dataclass with per-metric diagnostics
- `BurstTracker` — stateful event-duration estimator
- 31 tests

### Phase 5 — State Machine + Logging + CLI (THE PRODUCT)
- `runtime/state_machine.py` — idle → candidate → confirming → cooldown; gate agreement tracking; confirmation windows; refractory period (21 TDD tests)
- `runtime/logging.py` — append-only UTC-timestamped log; count persistence across invocations; size/age rotation
- `cli/main.py` — `start`, `view`, `reset`, `tail`, `status`, `purge` (all working)
- `requirements.txt` — `numpy>=2.0`, `sounddevice>=0.5`, `pytest>=9.0`
- `tests/fixtures/test_snorts.wav` — test fixture for dry-run

### Phase 6 — End-to-End Integration (v0.1)
- `runtime/classifier.py` — `Classifier` ABC + `GateHeuristicClassifier` placeholder (swap-ready for Phase 10)
- CLI refactored to use `create_classifier()` factory instead of hardcoded 0.85/0.10 probabilities
- `tests/test_integration.py` — 9 integration tests: full pipeline with WAV fixtures, logging, cooldown, config validation
- `## Privacy` section in README — data map, purge instructions, local-first guarantees
- `purge` CLI command (`-y`/`--yes` for automation) — wipes main log + rotated archives

### Phase 7 — Feature Extraction
- `runtime/features.py` — `FeatureExtractor` class: pure-NumPy log-Mel spectrogram
- `_mel_filterbank()` — triangular Mel filterbank builder (Hz↔Mel conversion)
- Config-validated at construction (n_fft, f_min, f_max, n_mels)
- `extract(audio)` → (n_mels,) float32; `extract_batch(segments)` → (batch, n_mels)
- 30 tests: Mel scale roundtrip, filterbank properties, determinism, silence, edge cases

### Council Fixes Applied
- Audio constants consolidated: `audio_contract.py` is the single source of truth; `AudioConfig` reads from it
- Gate optimized: RMS/ZCR computed on float32 (no upcast); early exit saves 30–50% on silent windows
- `BurstTracker` hardcoded `25.0` replaced with `WINDOW_SIZE_MS` from contract
- `requirements.txt` created
- Phases reordered per council consensus: product logic (state machine + CLI) built before ML pipeline

---

## 3. File Inventory

```
SnortTracker/
├── README.md                          # Project overview, setup, CLI usage
├── ROADMAP.md                         # Refined 13-phase roadmap
├── HANDOFF.md                         # This file
├── requirements.txt                   # numpy, sounddevice, pytest
├── .gitignore
├── LICENSE
│
├── runtime/                           # Live detection pipeline
│   ├── __init__.py
│   ├── config.py                      # Tuneable thresholds (Gate, Feature, Inference, StateMachine, Log)
│   ├── audio_contract.py              # Single source of truth for audio constants + validation
│   ├── capture.py                     # RingBuffer, MicrophoneCapture, WavFileCapture
│   ├── gating.py                      # RMS/ZCR/HF-ratio metrics, Gate, BurstTracker
│   ├── classifier.py                  # Classifier ABC + GateHeuristicClassifier placeholder
│   ├── features.py                    # Log-Mel spectrogram FeatureExtractor (pure NumPy)
│   ├── state_machine.py               # SnortState enum, StateMachine class (idle→candidate→confirming→cooldown)
│   └── logging.py                     # EventLogger — append-only log with rotation
│
├── cli/                               # Command-line interface
│   ├── __init__.py
│   └── main.py                        # start, view, reset, tail, status, purge
│
├── dataset/                           # Dataset tooling (empty — Phase 8)
│   └── __init__.py
│
├── train/                             # Model training (empty — Phase 9)
│   └── __init__.py
│
├── tests/                             # Test suite (161 tests)
│   ├── __init__.py
│   ├── test_runtime_smoke.py          # 4 tests — repo structure, config loading
│   ├── test_audio_contract.py         # 35 tests — validation, inspection, WAV, padding
│   ├── test_capture.py                # 22 tests — ring buffer, WAV capture, metadata
│   ├── test_gating.py                 # 31 tests — metrics, Gate, BurstTracker
│   ├── test_state_machine.py          # 21 tests — state transitions, cooldown, gate agreement
│   ├── test_integration.py            # 14 tests — full pipeline, classifier, config
│   ├── test_features.py               # 30 tests — Mel utils, filterbank, extraction, batch
│   └── fixtures/
│       └── test_snorts.wav            # 2-second test WAV with snort-like bursts
│
├── models/                            # Trained model artifacts (empty — Phase 10)
├── logs/                              # Event logs (gitignored)
│
├── Data/                              # MemorySmith wiki deployment (NOT a Python package)
│   └── Pages/
│       ├── Plans/                     # Design documents
│       ├── Tasks/                     # Phase trackers
│       └── Council/                   # Council review reports
│
└── .github/
    └── skills/
        └── council/
            └── SKILL.md               # Council review skill
```

---

## 4. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `dataset/` not `data/` | Windows case-insensitive collision with `Data/` (MemorySmith wiki) |
| `audio_contract.py` is single source of truth | Prevents silent preprocessing drift between training and inference |
| Gate uses AND logic (all metrics must agree) | Conservative by design — gate exists to save compute, not define correctness |
| `is_positive` is classifier-only; gate tracked separately | Classifier is the primary signal; gate agreement is an independent safety check |
| State machine built before ML pipeline | Council consensus: product first, model accuracy second |
| `WavFileCapture` dry-run mode | Enables full pipeline testing without hardware |
| All data local, no network code | Privacy-first; no telemetry, no cloud, no sync |
| `Classifier` abstract base + factory | Clean swap point — `GateHeuristicClassifier` today, ONNX model in Phase 10 |
| Feature extraction shared between runtime and training | `FeatureExtractor` in `runtime/` imported by training; prevents preprocessing drift |
| Pure NumPy feature extraction | No librosa/scipy dependency at inference; keeps Pi Zero deployment lightweight |

---

## 5. How to Run

```bash
# Setup
git clone https://github.com/TheMasonX/SnortTracker.git
cd SnortTracker
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# Run tests
python -m pytest tests/         # 113 tests

# Dry-run with a WAV file
python -m cli.main start --input path/to/recording.wav

# View results
python -m cli.main view         # Show count
python -m cli.main tail -n 10   # Show last 10 events
python -m cli.main status       # Show runtime health
python -m cli.main reset        # Clear logs and count
```

---

## 6. What's Next (Per ROADMAP.md)

| Phase | What | Why |
|-------|------|-----|
| **8** | Dataset collection | Record real snorts + hard negatives; label policy (positive/negative/ignore) |
| **9** | Model training | Tiny CNN, trained on desktop GPU, event-level evaluation |
| **10** | Model swap & comparison | Export to ONNX, swap into runtime via `create_classifier()`, compare counts against v0.1 baseline |
| **11** | Pi Zero optimization | Profile ONNX latency, eliminate remaining allocations, GC profiling |
| **12** | Calibration & tuning | `calibrate` CLI, threshold persistence, environment-specific defaults |
| **13** | Hardening | Soak tests, regression fixtures, edge case coverage, Pi Zero smoke test |

---

## 7. Important Context for Next Developer

### Windows-specific
- The `Data/` folder (capital D) is NOT a Python package — it's for MemorySmith wiki deployment
- The Python dataset package is `dataset/` (lowercase) to avoid case-insensitive collision
- All commands use `.venv\Scripts\activate` on Windows; `source .venv/bin/activate` on Unix

### Virtual environment
- `.venv/` is gitignored; recreate with `python -m venv .venv && pip install -r requirements.txt`
- Dependencies: `numpy`, `sounddevice`, `pytest` (and their transitive deps)

### State machine semantics
- `is_positive` depends ONLY on `probability >= threshold` (classifier)
- `gate_passed` is tracked independently via `gate_agreement_count`
- `min_gate_agreement` requires the gate to agree on at least N confirmation windows
- `confirmation_windows=1` with `min_gate_agreement=1` counts on the first positive (useful for testing)

### Placeholder classifier
- `runtime/classifier.py` defines `Classifier` (ABC) and `GateHeuristicClassifier` (v0.1 placeholder)
- The CLI uses `create_classifier(config.inference)` factory — returns `GateHeuristicClassifier` today
- `GateHeuristicClassifier` maps gate pass/fail + RMS energy → probability:
  - Gate fail → probability=0.0
  - Gate pass → `base_prob + rms_boost + hf_boost`, clamped to `max_prob`
- In Phase 10: `create_classifier()` detects ONNX model path and returns `ONNXClassifier`
- The interface is `predict(audio, gate_result) -> float` — both stats and future models use this

### Test fixtures
- `tests/fixtures/test_snorts.wav` is a generated 2-second 16kHz mono WAV with snort-like 3kHz bursts
- Regenerate with numpy if needed; NOT committed to git (should be in `.gitignore` — verify)

### Log files
- `logs/` is gitignored
- Log format: `ISO8601_UTC, event_id=N, confidence=X.XXXX, model=NAME, config=VERSION`
- Count persists by counting lines in the log file on init
- Rotation: 30 days max age, 10 MB max size (best-effort)

---

## 8. Known Gaps & Technical Debt

| Gap | Severity | Notes |
|-----|----------|-------|
| No live microphone tested | Medium | `MicrophoneCapture` exists but only WAV dry-run tested |
| Placeholder classifier | Medium | `GateHeuristicClassifier` is gate-based; real CNN model needed (Phase 10) |
| No model training data | Critical | Zero recordings of actual snorts |
| Config validation | Low | No bounds checking on negative cooldown, >1.0 probability |
| `zero_pad_window` always copies | Low | Perf Engineer flagged; exact-size windows copy unnecessarily |
| `datetime.now()` per window | Low | Syscall on every 10ms hop; should use monotonic counter + periodic sync |
| No soak test | Medium | No test runs the pipeline for >2 seconds |
| `tests/fixtures/` not gitignored | Medium | Guardian flagged; real snort WAVs placed here would be committed |
| `FeatureExtractor` not yet wired into CLI | Medium | Features are extractable but CLI pipeline still uses gate-only path (expected until Phase 10) |

---

## 9. Council Review Summary (2026-06-28)

Six LLM reviewers audited Phases 1–4. Key findings:

- **Architecture: 7/10** — Clean linear pipeline, good abstractions, duplicate constants was the main flaw (fixed)
- **Shippability: 4/10** — Product logic was 7 phases away; reordered to Phase 5 (fixed)
- **Test Coverage: 3/10** — Good unit tests but no integration, state machine, or end-to-end tests (fixed — TDD state machine, integration tests, feature tests: 161 total)
- **Privacy: 5/10** — Local-first enforced but no rotation, no purge, no privacy docs (all fixed — rotation, purge CLI, `## Privacy` in README)
- **Performance: 4/10** — Float64 upcasts and allocation churn were hot-path issues (fixed — float32 metrics, early exit)
- **Usability: 3/10** — No `start` command, nothing user-facing (fixed — full CLI with 6 commands)

**Current rating: ~8/10** — Product complete, feature extraction ready, well-tested pipeline. Ready for dataset collection and model training.

---

## 10. Quick Verification

```bash
# Everything should pass
python -m pytest tests/ -q
# Expected: 161 passed in ~4s

# Dry-run should count at least 1 snort
python -m cli.main start --input tests/fixtures/test_snorts.wav
# Expected: "Snorts counted: 1"

# Feature extraction smoke test
python -c "from runtime.features import FeatureExtractor; import numpy as np; fe=FeatureExtractor(); print(fe.extract(np.random.randn(400).astype('float32')).shape)"
# Expected: (40,)
```
