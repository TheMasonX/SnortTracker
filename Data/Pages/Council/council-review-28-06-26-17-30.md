# 🏛️ SnortTracker — LLM Council Review

**Date:** 2026-06-28 17:30 UTC  
**Review scope:** Phases 1–4 (complete) + full design documents  
**Council composition:** 6 reviewers with distinct perspectives  

---

## Council Members

| Role | Focus | Rating |
|------|-------|--------|
| 🏗️ The Architect | Architecture, abstractions, coupling | 7/10 |
| 🔨 The Pragmatist | Scope, shippability, timeline | 4/10 |
| 🧪 The Tester | Test coverage, edge cases, quality | 3/10 |
| 🛡️ The Guardian | Privacy, data hygiene, local-first | 5/10 |
| 🔬 The Perf Engineer | Pi Zero runtime, allocations, GC | 4/10 |
| 🧑‍🦱 The User Advocate | UX, CLI, real-world usability | 3/10 |

**Average rating:** 4.3 / 10

---

## Unanimous Findings (All 6 Agree)

### 🔴 #1: Build the Product Before the ML Pipeline

**Filed by:** Pragmatist, User Advocate, Architect, Tester  

The current plan has Phases 5–8 (dataset pipeline, training, export) before Phases 9–11 (state machine, logging, CLI). This means the product — a working snort counter — is 5+ phases away. Meanwhile, capture and gating already work end-to-end.

**Consensus:** Reorder to build state machine + logging + CLI next. Wire them into capture + gate with a dummy classifier. Ship v0.1 that counts *something*. Then iterate on the model.

### 🔴 #2: Duplicate Audio Constants Are a Time Bomb

**Filed by:** Architect, Perf Engineer, Tester  

`audio_contract.py` and `config.py` independently define `SAMPLE_RATE`, `WINDOW_SIZE_MS`, `HOP_SIZE_MS`, `WINDOW_SAMPLES`, `HOP_SAMPLES`, and `DTYPE`. `capture.py` imports from the contract; `gating.py` imports from both. A change in one place creates silent drift.

**Consensus:** Make `audio_contract.py` the single source of truth. Either delete `AudioConfig` or have it read from the contract constants.

### 🔴 #3: Missing `requirements.txt`

**Filed by:** Pragmatist, User Advocate  

The README references `pip install -r requirements.txt` but no such file exists. Dependencies (`numpy`, `sounddevice`, `pytest`) are installed ad hoc. The project is not reproducible.

**Consensus:** Create `requirements.txt` immediately.

### 🔴 #4: `BurstTracker` Hardcodes Window Duration

**Filed by:** Architect, Tester, Perf Engineer  

`BurstTracker.current_duration_ms` returns `self._consecutive_passes * 25.0` — a magic number. The `update()` method accepts a `window_duration_ms` parameter but ignores it. If `WINDOW_SIZE_MS` changes, burst tracking silently reports wrong durations.

**Consensus:** Replace `25.0` with `WINDOW_SIZE_MS` from the contract.

---

## Per-Member Findings

### 🏗️ The Architect — Rating: 7/10

**Strengths:** Clean linear pipeline, excellent design docs, right abstractions at the right level, `WavFileCapture` dry-run is a smart design decision, gating AND logic is correct.

**Issues:**
- `FeatureConfig` has no validation against contract (`f_max ≤ Nyquist`, `hop_length == HOP_SAMPLES`)
- `AudioConfig` duplicates contract constants
- State machine should be built before dataset pipeline

**MUST-FIX:** Consolidate audio constants — `audio_contract.py` becomes the single source of truth.

---

### 🔨 The Pragmatist — Rating: 4/10

**Strengths:** Code quality is high, test discipline is good, gate-first design is correct.

**Issues:**
- 40% of planned infrastructure is premature (dataset manifest format, export pipeline, augmentation framework)
- Zero training data exists — every ML phase is gated on nonexistent recordings
- No end-to-end test proves the pipeline works when connected
- `requirements.txt` is literally commented out

**MUST-FIX:** Record 10 minutes of audio TODAY. Snorts, silence, speech, coughs. Without data, the project is a beautifully engineered empty box.

---

### 🧪 The Tester — Rating: 3/10

**Strengths:** Existing unit tests are well-written, use pytest idioms, test behavior not internals. `validate_window` and ring buffer tests are thorough.

**Issues:**
- **No state machine tests** (module doesn't exist yet)
- **No integration tests** — every component tested in isolation
- **No inference, logging, or CLI tests** (modules don't exist)
- `test_runtime_smoke.py` is not a smoke test — it's a directory existence check
- `BurstTracker` tests are coupled to hardcoded `25.0` constant
- 12 edge cases completely uncovered (NaN to gate, zero-capacity ring buffer, 24-bit WAV, negative config values...)

**MUST-FIX:** Write `test_state_machine.py` BEFORE `runtime/state_machine.py`. TDD the correctness layer.

---

### 🛡️ The Guardian — Rating: 5/10

**Strengths:** No network code anywhere, local-only commitment is real in practice, `.gitignore` coverage is good, design docs correctly identify biometric risks.

**Issues:**
- **Log data is biometric surveillance** — UTC-timestamped snort events with confidence scores constitute a sleep-pattern time series with zero rotation, zero TTL, zero purge mechanism
- **No encryption-at-rest** for audio recordings on the Pi's SD card
- **No documented threat model** — if someone steals the Pi, they have identifiable biometric data
- **stderr leak in gate debug mode** — gate metrics print to stderr unconditionally when debug is on
- `.gitignore` gaps: `*.csv`, `*.pkl`, `*.db`, `tests/fixtures/` not ignored
- **No PRIVACY.md** — zero documentation of what data is stored and for how long

**MUST-FIX:** Add log rotation (`max_age_days`, `max_size_mb`), a `purge` CLI command, and a `## Privacy` section in README.

---

### 🔬 The Perf Engineer — Rating: 4/10

**Strengths:** Pipeline architecture is right for constrained hardware. Gate-first design will save massive compute once inference is connected. Ring buffer is well-sized.

**Issues:**
- **Triple float64 upcast in gate** — `rms_energy`, `zero_crossing_rate`, and `high_freq_energy_ratio` each independently convert `float32` → `float64`, allocating 3.2 KB each. At 100 windows/sec, that's **960 KB/sec of temporary allocations**.
- **No early exit in gate** — if RMS fails (most windows), ZCR and FFT still run
- **`zero_pad_window()` always copies** even for exact-size windows (common case)
- **`datetime.now()` per window** is a syscall on every 10ms hop
- **`GateResult` dataclass created and GC'd** even in the `passes()` convenience shortcut
- **GC world-stop risk** — 1,000–1,500 allocs/sec will trigger gen2 collections causing 50–200ms pauses that block the sounddevice callback → silent buffer overruns
- **ONNX inference latency unknown** — 50–200ms per forward pass on ARMv6 could collapse real-time pipeline if gate pass rate is too high
- **No profiling on Pi Zero yet** — all performance discussion is theoretical

**MUST-FIX:** Eliminate triple float64 upcast. Compute RMS and ZCR directly on float32; only upcast for the FFT. Saves ~640 KB/sec of allocation churn.

---

### 🧑‍🦱 The User Advocate — Rating: 3/10

**Strengths:** Concept is genuinely clever ("step counter for snorts"). Planned CLI commands (view, reset, tail, status, calibrate) are well-chosen. Plaintext UTC-timestamped logs are user-friendly.

**Issues:**
- **No `start`/`run`/`listen` command** — the five documented CLI commands are management tools for a process with no documented way to start
- **No user-facing functionality exists** — zero CLI, zero counter, zero output. Purely developer infrastructure.
- **On-ramp requires terminal comfort** — acceptable for Pi MVP, but no `requirements.txt` makes it actively hostile
- **No audio replay for debugging** — if the count is wrong, the user can't hear what the system heard
- **Gate metrics missing from event logs** — the log line omits gate pass/fail context

**MUST-FIX:** Implement `cli/main.py` with a `start` subcommand that wires capture + gate into a placeholder counter and writes to a log file. Close the loop TODAY.

---

## Consolidated MUST-FIX List (Priority Order)

| # | Severity | Item | Owner |
|---|----------|------|-------|
| 1 | 🔴 | **Consolidate audio constants** — `audio_contract.py` is single source of truth | Architect |
| 2 | 🔴 | **Reorder phases** — build state machine + logging + CLI before ML pipeline | Pragmatist |
| 3 | 🔴 | **Create `requirements.txt`** | Pragmatist |
| 4 | 🔴 | **Fix `BurstTracker` hardcoded `25.0`** → use `WINDOW_SIZE_MS` | Architect |
| 5 | 🔴 | **Eliminate triple float64 upcast in gate** | Perf Engineer |
| 6 | 🔴 | **Add log rotation + `purge` CLI command** | Guardian |
| 7 | 🔴 | **Write `test_state_machine.py` before `state_machine.py`** | Tester |
| 8 | 🔴 | **Record real audio data** | Pragmatist |
| 9 | 🔴 | **Add `start` CLI command + close the end-to-end loop** | User Advocate |
| 10 | 🟡 | **Add `PRIVACY.md` or `## Privacy` section to README** | Guardian |
| 11 | 🟡 | **Add early exit to gate** (short-circuit on RMS fail) | Perf Engineer |
| 12 | 🟡 | **Replace `test_runtime_smoke.py` with real pipeline smoke test** | Tester |
| 13 | 🟡 | **Add `FeatureConfig` contract validation** | Architect |
| 14 | 🟡 | **Add `Config` value validation** (reject negative thresholds, >1.0 probabilities) | Tester |
| 15 | 🟡 | **Profile ONNX inference on Pi Zero** before committing to model architecture | Perf Engineer |

---

## Revised Phase Order (Council Consensus)

```
Phase 1–4:  ✅ DONE  — skeleton, contract, capture, gating
Phase 5:    🔜 NEXT — state machine + logging + CLI (THE PRODUCT)
Phase 6:    🔜 NEXT — wire end-to-end with dummy classifier, ship v0.1
Phase 7:             — record real data (gold positives + hard negatives)
Phase 8:             — feature extraction (shared contract)
Phase 9:             — train tiny model
Phase 10:            — swap real model in, compare against baseline
Phase 11:            — export, quantize, optimize for Pi Zero
Phase 12:            — calibration + threshold tuning
Phase 13:            — hardening, fixtures, edge cases
```

**Key change:** The product (state machine + logging + CLI) is now Phase 5–6 instead of 9–11. The ML pipeline (dataset → train → export) moves to 7–11. This means a working prototype in 2 phases instead of 7.

---

## Risk Register Update

| Risk | Pre-Review | Post-Review | Mitigation |
|------|-----------|-------------|------------|
| Label noise | Medium | Medium | Keep ignore bucket; unchanged |
| Double-counting | Medium | **High** | State machine not built yet — accelerate |
| Over-fragmented codebase | Low | Low | Current abstraction level is right |
| Preprocessing drift | Medium | **High** | Duplicate constants found — fix now |
| Device variability | Low | Medium | Calibration is important, not urgent |
| **NEW: GC-induced underruns** | — | **High** | Allocation churn in gate hot path |
| **NEW: Biometric log leakage** | — | **Medium** | No rotation, no purge, no privacy doc |
| **NEW: No training data** | — | **Critical** | Zero recordings exist |
| **NEW: ONNX latency unknown** | — | **High** | Untested on target hardware |

---

## Verdict

The council finds the SnortTracker project to have **strong foundations** (clean architecture, good tests, excellent design docs) but **critical sequencing problems** (product logic after ML pipeline, no training data, no end-to-end loop). 

The unanimous recommendation: **reorder to build the counter first, then train the model.** A working v0.1 shipped next week beats a perfect v1.0 shipped never.

---

*Council adjourned 2026-06-28 17:30 UTC. Six members, six perspectives, one consensus.*
