# Phase 5 — State Machine + Logging + CLI — Task Tracker

**Status:** ✅ Complete  
**Started:** 2026-06-28  
**Completed:** 2026-06-28  
**Goal:** Build the counting logic and user interface — the actual product.

---

## Task List

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Create `requirements.txt` | ✅ Done | `numpy>=2.0`, `sounddevice>=0.5`, `pytest>=9.0` |
| 2 | Consolidate audio constants | ✅ Done | `audio_contract.py` is single source of truth; `AudioConfig` imports from it |
| 3 | Fix `BurstTracker` hardcoded `25.0` | ✅ Done | Now uses `WINDOW_SIZE_MS` from contract |
| 4 | Optimize gate: early exit | ✅ Done | RMS → ZCR → HF ratio evaluated sequentially; short-circuits on fail |
| 5 | Optimize gate: float32 metrics | ✅ Done | RMS and ZCR computed on float32 (no upcast); only HF ratio upcasts for FFT |
| 6 | State machine: TDD tests | ✅ Done | `tests/test_state_machine.py` — 21 tests, written before implementation |
| 7 | State machine: implementation | ✅ Done | `runtime/state_machine.py` — idle → candidate → confirming → cooldown |
| 8 | Logging: implementation | ✅ Done | `runtime/logging.py` — append-only, UTC timestamps, rotation, count persistence |
| 9 | CLI: `cli/main.py` | ✅ Done | `start`, `view`, `reset`, `tail`, `status` — all working |
| 10 | End-to-end dry-run test | ✅ Done | Processed test WAV: 80 windows, 50 gate passes, 1 snort counted |
| 11 | Run full test suite | ✅ Done | 113/113 pass |

---

## Exit Criteria

- [x] State machine correctly transitions through all 5 states
- [x] State machine prevents double-counting via cooldown
- [x] Gate agreement is tracked independently of classifier probability
- [x] Log events are append-only with UTC timestamps
- [x] Log count persists across CLI invocations
- [x] `reset` clears logs and count
- [x] CLI `start` works with `--input` WAV dry-run mode
- [x] CLI `view`, `tail`, `status` report correctly

---

## Council MUST-FIX Items Addressed

| # | Item | Status |
|---|------|--------|
| 1 | Consolidate audio constants | ✅ |
| 2 | Reorder phases (product before ML) | ✅ This IS the new Phase 5 |
| 3 | Create `requirements.txt` | ✅ |
| 4 | Fix `BurstTracker` hardcoded `25.0` | ✅ |
| 5 | Eliminate triple float64 upcast | ✅ |
| 6 | Build state machine before dataset pipeline | ✅ |
| 7 | Write state machine tests before implementation | ✅ |
| 9 | Add `start` CLI command + close loop | ✅ |

## Files Created

| File | Purpose |
|------|---------|
| `requirements.txt` | Frozen dependencies |
| `runtime/state_machine.py` | Event confirmation state machine |
| `runtime/logging.py` | Append-only event logger with rotation |
| `cli/main.py` | CLI with start/view/reset/tail/status |
| `tests/test_state_machine.py` | 21 TDD tests for the state machine |
| `tests/fixtures/test_snorts.wav` | Test fixture for dry-run |

## Test Results

- **113 passed in 1.20s** (4 smoke + 35 contract + 22 capture + 31 gating + 21 state machine)
- State machine correctly counts 1 snort from a 2-second WAV with 50 gate-passing windows

## Notes

- State machine uses `is_positive = probability >= threshold` (classifier-only); gate agreement is tracked independently
- `min_gate_agreement` ensures the gate must agree on at least N windows during confirmation
- `confirmation_windows=1` with `min_gate_agreement=1` counts on the first positive window — useful for initial testing
- CLI dry-run mode (`--input`) processes WAV files without hardware
- Log rotation: max 30 days age, 10 MB size — best-effort rotation to `.YYYYMMDDTHHMMSS.log`