# Phase 4 — Cheap Gating — Task Tracker

**Status:** ✅ Complete  
**Started:** 2026-06-28  
**Completed:** 2026-06-28  
**Goal:** Reject obvious non-events before neural inference is called.

---

## Task List

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Implement RMS energy metric | ✅ Done | `rms_energy()` — root-mean-square amplitude |
| 2 | Implement high-frequency energy ratio | ✅ Done | `high_freq_energy_ratio()` — FFT-based, configurable cutoff |
| 3 | Implement zero-crossing rate | ✅ Done | `zero_crossing_rate()` — proportion of sign changes |
| 4 | Implement `GateResult` dataclass | ✅ Done | pass/fail + per-metric values + reason string |
| 5 | Implement `Gate` class | ✅ Done | Combines metrics, configurable thresholds, `evaluate()`, `passes()`, debug mode |
| 6 | Add event-duration tracking | ✅ Done | `BurstTracker` — stateful consecutive-pass counter with min/max duration gates |
| 7 | Make thresholds configurable from `GateConfig` | ✅ Done | Wired to `runtime.config.GateConfig` |
| 8 | Write gating tests | ✅ Done | `tests/test_gating.py` — 31 tests |
| 9 | Run tests and confirm pass | ✅ Done | 92/92 pass (combined across all test files) |

---

## Exit Criteria

- [x] Gate rejects obvious negatives (silence, low-energy noise)
- [x] Gate passes candidate snort-like bursts
- [x] Each metric is individually testable
- [x] Debug mode emits per-metric values
- [x] Gate is conservative — does not become the final decision-maker

---

## Files Created

| File | Purpose |
|------|---------|
| `runtime/gating.py` | `rms_energy()`, `zero_crossing_rate()`, `high_freq_energy_ratio()`, `GateResult`, `Gate`, `BurstTracker` |
| `tests/test_gating.py` | 31 tests covering all three metrics, Gate, and BurstTracker |

## Notes

- Gate combines three metrics with AND logic — all must agree, keeping it conservative
- `GateResult` carries per-metric pass/fail booleans for diagnostics
- `BurstTracker` is a stateful companion that estimates event duration across consecutive passes (helps reject too-short transients and too-long noise)
- Debug mode prints to stderr when enabled
- All thresholds pulled from `GateConfig` defaults (RMS ≥ 0.01, ZCR ∈ [0.02, 0.40], HF ratio ≥ 0.15)
- Combined test run: **92 passed in 0.93s** (4 smoke + 35 contract + 22 capture + 31 gating)
