# Snort Counter System — Initial Design Document
## Clean Synthesis of Requirements, Audits, and Implementation Direction

**Status:** Initial design baseline  
**Purpose:** Establish the simplest version of the system that is still reliable, testable, portable, and easy to evolve later.

---

## 1) Product Goal

Build a playful snort counter that behaves like a step counter: it continuously listens, detects **full snorts**, and increments a local count when a snort is confidently confirmed.

This is not intended to be a general audio understanding system. It is a narrow event detector for one specific use case. The system should be directional and useful, not perfect.

### Success criteria
The MVP succeeds if it can:

- reliably count full snorts in the target user’s normal environment,
- avoid double-counting one snort as multiple events,
- stay small enough to run on a Pi-class device,
- remain easy to test with recorded audio fixtures,
- keep all data local for the MVP.

---

## 2) Human Requirements

### 2.1 Core functional goal
- Detect full snorts over time.
- Count them like step events.
- Keep the experience lightweight and low-friction.

### 2.2 Detection requirements
- Detect **full snorts only**.
- Do not count partial snorts, quiet nasal bursts, or ambiguous sounds.
- Use a binary decision: **snort / not snort**.
- Attach a confidence score to each model output.
- Bias training toward the target user’s snorts without building a biometric identity system.

### 2.3 Runtime and deployment requirements
- Primary MVP deployment target: Raspberry Pi Zero-class hardware with a USB microphone.
- Continuous listening is required.
- Use cheap gating before neural inference.
- Keep the model tiny and efficient.
- Use a short confirmation window before counting.
- Store counts locally only.
- Initial interface: CLI and log file only.

### 2.4 Dataset and training requirements
- Use the target user’s snorts as the most important positive data.
- Supplement with the assistant’s snorts, downloaded samples, and auto-mined candidates.
- Prefer automation for dataset collection and processing.
- Use strong negative coverage: speech, laughter, breathing, sniffing, throat clears, coughs, ambient noise, and silence.
- Train off-device on a desktop GPU.
- Use a small, high-quality dataset rather than a large noisy one.

### 2.5 Explicit non-goals for MVP
- No dashboard.
- No sync.
- No mobile app.
- No multi-class classifier.
- No emotional context detection.
- No speaker-identification or “her-only” biometric model.
- No partial-snort counting.

---

## 3) Design Principles

These decisions are the main synthesis of the audits and feedback.

### 3.1 Keep the runtime small
The runtime should be one simple pipeline, not a sprawling graph of loosely connected modules. The essential flow is:

**capture → cheap prefilter → classifier → event state machine → append-only log**

That is the core of the system. Everything else supports it.

### 3.2 Treat detection as a state machine
The confirmation logic should be deterministic, explicit, and testable. A snort counter is not just a threshold checker. It needs:
- candidate detection,
- confirmation,
- count emission,
- cooldown / refractory handling.

### 3.3 Treat labels as the real bottleneck
Model size is not the hard part. Label quality is. The system should preserve an explicit **ignore / uncertain** path in labeling, even though the model output is binary.

### 3.4 Use one shared preprocessing contract
Training and inference must use the same feature pipeline. Preprocessing is part of the model contract, not a side utility.

### 3.5 Prefer portable inference
The deployment stack should stay simple and portable. Use a quantized model and a runtime that is easy to keep stable on the target device.

### 3.6 Measure what matters
Measure event-level performance, not just window-level accuracy. The meaningful question is whether the system counts real snorts correctly in use.

### 3.7 Add calibration and health checks
Each device should be able to calibrate its thresholds against the local environment. The runtime should also expose basic health signals such as dropped windows, buffer underflow, and model load failures.

---

## 4) High-Level System Architecture

### 4.1 Runtime pipeline
1. Audio capture
2. Cheap gating / prefiltering
3. Feature extraction
4. Tiny binary classifier
5. Confirmation state machine
6. Count emission
7. Local logging

### 4.2 Supporting toolchain
Separate tools should handle:
- dataset slicing,
- auto-mining candidates,
- labeling support,
- feature generation,
- model training,
- model export,
- offline evaluation,
- fixture-based tests.

### 4.3 Suggested repo structure

```text
snort-counter/
├── runtime/
│   ├── capture.py
│   ├── gate.py
│   ├── features.py
│   ├── inference.py
│   ├── state_machine.py
│   ├── logging.py
│   └── app.py
├── dataset/
│   ├── collection/
│   ├── slicing/
│   ├── labeling/
│   └── augmentation/
├── train/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── export.py
├── cli/
│   └── main.py
├── tests/
│   ├── fixtures/
│   ├── test_gate.py
│   ├── test_state_machine.py
│   ├── test_logging.py
│   └── test_runtime_smoke.py
├── models/
├── logs/
└── README.md
```

> **Note:** The `Data/` directory at the repo root is reserved for the MemorySmith wiki
> deployment layer. Python source packages live alongside it (e.g., `dataset/`, `runtime/`).
> All Python work uses a virtual environment (`.venv/`) for portability and reproducibility.

This keeps the codebase simple while still separating runtime, dataset, training, and tests.

---

## 5) Runtime Design

## 5.1 Audio capture
**Responsibility:** Continuously read microphone input and provide fixed-size windows to the rest of the pipeline.

### Required behavior
- 16 kHz mono input.
- Stable streaming with a ring buffer.
- Configurable window size.
- Explicit handling for underflow, overflow, and device errors.

### Preferred API
```python
def get_audio_window() -> np.ndarray:
    """Return the next audio window as a 1D float32 array."""
```

### Practical notes
- The capture layer should be stateful.
- It should expose a small health object or metadata alongside audio.
- It should not silently drop samples.

---

## 5.2 Gating / prefiltering
**Responsibility:** Reject obvious non-events before expensive inference runs.

### Required behavior
- Use cheap features only.
- Keep the gate conservative.
- Log gate decisions for later analysis.
- Do not let the gate become the final decision-maker.

### Suggested signals
- RMS energy
- high-frequency energy
- zero-crossing rate
- rough duration / burst shape

### Preferred API
```python
def passes_gate(audio_window: np.ndarray) -> bool:
    """Return True if the window should be sent to inference."""
```

### Design rule
The gate exists to reduce compute, not to define correctness.

---

## 5.3 Feature extraction
**Responsibility:** Convert audio into model-ready features.

### Required behavior
- Use the same feature pipeline during training and inference.
- Produce a compact representation suitable for a tiny CNN.
- Keep the implementation deterministic and versioned.

### Suggested feature type
- log-Mel spectrogram
- compact mel bin count
- fixed window length

### Design rule
If the feature pipeline changes, it is effectively a model change and must be versioned as such.

---

## 5.4 Model inference
**Responsibility:** Return a binary snort probability.

### Required behavior
- Load a tiny quantized model.
- Return a single probability in [0, 1].
- Be fast enough for the chosen device.
- Fail clearly if the model cannot load.

### Preferred API
```python
def infer_snort_probability(features: np.ndarray) -> float:
    """Return the model confidence that the event is a snort."""
```

### Model shape
A minimal CNN is sufficient:
- convolution blocks,
- lightweight pooling,
- one dense output,
- sigmoid probability.

The model should be small, portable, and easy to export.

---

## 5.5 Event confirmation state machine
**Responsibility:** Decide when a real snort event has happened.

This is one of the most important pieces in the system.

### Required states
- `idle`
- `candidate`
- `confirming`
- `cooldown`

### Required behavior
- Require sustained evidence across consecutive windows.
- Require agreement between gate and classifier.
- Emit exactly one count per snort event.
- Prevent tail fragments from being counted twice.
- Merge windows that belong to the same physical snort.

### Preferred API
```python
def update(probability: float, gate_passed: bool) -> bool:
    """
    Update state with the latest inference result.
    Return True when a new snort should be counted.
    """
```

### Implementation notes
- Use a short confirmation window.
- Add a refractory interval after a count.
- Keep state transitions explicit and unit-testable.

### Design rule
The state machine is the authority on event emission. The model only supplies evidence.

---

## 5.6 Logging and count storage
**Responsibility:** Persist snort events locally.

### Required behavior
- Append-only local log for MVP.
- UTC timestamps.
- Confidence recorded with each event.
- Stable, easy-to-parse format.

### Suggested log line
```text
2026-06-28T00:31:12Z, event_id=42, confidence=0.92, model=snort-cnn-v1
```

### Optional future fields
- gate summary
- config version
- runtime health metadata

### Design rule
The log format should be easy to inspect by hand and easy to migrate later.

---

## 6) Dataset Strategy

## 6.1 Positive data
The positive class should be built in layers:

1. **Gold positives**  
   Manually confirmed full snorts from the target user.

2. **Silver positives**  
   Assisted or downloaded candidates that have been reviewed and accepted.

3. **Auto-mined candidates**  
   Gated snippets from long recordings that still require confirmation before becoming training truth.

### Important rule
Auto-mined clips are **candidates**, not labels.

### Suggested handling
- Keep partial snorts and ambiguous nasal bursts out of the gold set.
- Use sample weighting to bias toward the target user’s snorts.
- Avoid letting weak positives contaminate validation data.

---

## 6.2 Negative data
Negative coverage should include:
- speech,
- laughter,
- breathing,
- sniffing,
- throat clears,
- coughs,
- ambient noise,
- silence.

These should include both easy negatives and hard negatives that acoustically resemble snorts.

---

## 6.3 Label policy
The annotation workflow should include three buckets:
- positive,
- negative,
- ignore / uncertain.

Even though the model is binary, the labeling process should not be forced into binary. Ambiguous clips should usually be excluded rather than mislabeled.

---

## 6.4 Dataset workflow
1. Record long sessions.
2. Use gating to mine candidates.
3. Manually confirm true positives.
4. Slice negatives from non-event segments.
5. Normalize audio to the standard sample rate.
6. Extract features.
7. Version the dataset manifest.
8. Split by event/session so leakage does not occur.

---

## 7) Training and Model Export

### 7.1 Training objective
Train a binary classifier with a loss suitable for class imbalance, and weight the target user’s positives more heavily than non-target positives.

### 7.2 Training rules
- Keep training and inference preprocessing identical.
- Validate on held-out sessions, not just random windows.
- Evaluate event-level performance.
- Do not allow auto-mined candidates into validation unless manually verified.

### 7.3 Export
After training:
- export the model,
- quantize it,
- package it with version metadata,
- keep preprocessing versioned alongside it.

### 7.4 Key design decision
The inference stack should be chosen for stability and portability first. Performance tuning comes second, after profiling on the actual device.

---

## 8) CLI and Calibration

### 8.1 CLI commands
The initial CLI should stay small:

```text
snort-counter view
snort-counter reset
snort-counter tail
snort-counter calibrate
snort-counter status
```

### 8.2 Calibration mode
Calibration should measure the local acoustic environment and produce default thresholds for:
- gate sensitivity,
- probability threshold,
- confirmation duration,
- refractory period.

### 8.3 Why calibration matters
A fixed threshold set will be brittle across rooms, microphones, and background noise. A short calibration phase makes the MVP more portable and less surprising.

---

## 9) Testing Strategy

Testing should prove correctness at three levels.

### 9.1 Unit tests
- gate behavior,
- state machine transitions,
- logging format,
- reset semantics,
- calibration output handling.

### 9.2 Fixture-based offline tests
Use recorded WAV clips to validate:
- no double-count on one long snort,
- no count on speech or laughter,
- no count on ambiguous nasal bursts,
- correct handling of silence and noise,
- correct recovery from empty or noisy input.

### 9.3 Hardware smoke test
Run the pipeline on the target device with a live microphone to confirm:
- capture works,
- gating runs,
- inference loads,
- counts are emitted,
- logs are durable.

### 9.4 Acceptance criteria
A useful MVP should:
- count obvious full snorts correctly,
- avoid repeated counts on one event,
- remain stable during continuous listening,
- keep enough observability to debug false positives and false negatives.

---

## 10) Phased Implementation Plan

### Phase 1 — Skeleton and contracts
- Create the repo structure.
- Define the runtime APIs.
- Define the event schema.
- Define the dataset manifest format.
- Add basic tests.

### Phase 2 — Audio capture and gating
- Implement continuous capture.
- Add ring buffer handling.
- Implement the cheap gate.
- Log gate decisions for debugging.

### Phase 3 — Dataset tooling
- Build slicing and labeling helpers.
- Add candidate mining support.
- Add feature generation.
- Build a small, curated initial dataset.

### Phase 4 — Model training
- Implement the tiny CNN.
- Train and validate on curated data.
- Tune thresholds against event-level metrics.
- Export a quantized inference artifact.

### Phase 5 — Runtime integration
- Connect capture, gate, inference, and state machine.
- Write events to local logs.
- Run live tests on the target device.

### Phase 6 — CLI and calibration
- Add view, reset, tail, status, and calibrate.
- Validate threshold persistence.
- Confirm reset and log behavior.

### Phase 7 — Hardening
- Add more fixtures.
- Improve edge-case handling.
- Tighten data hygiene.
- Reduce false positives and double-counting.

---

## 11) Risks and Mitigations

### Risk: label noise
**Mitigation:** Keep an explicit ignore bucket and require human confirmation of mined positives.

### Risk: double-counting
**Mitigation:** Use a state machine with cooldown and event merging.

### Risk: over-fragmented codebase
**Mitigation:** Keep the runtime simple and prefer one package over many tiny boundaries.

### Risk: preprocessing drift
**Mitigation:** Share preprocessing logic or version it together with the model.

### Risk: device variability
**Mitigation:** Add calibration and health reporting.

### Risk: unrealistic hardware expectations
**Mitigation:** Verify early on the actual Pi model and microphone combination.

---

## 12) Open Decisions to Resolve Next

- Exact board target: original Pi Zero or a faster Zero-class variant.
- Final feature representation.
- Confirmation duration and refractory timing.
- Logging schema details.
- Calibration persistence format.
- Minimum gold-positive dataset size before training begins.
- Whether the first runtime uses pure Python orchestration with a compact native inference layer.

---

## 13) Recommended Final Shape for the MVP

If this is kept intentionally small, the best MVP is:

- a single runtime pipeline,
- a conservative gate,
- a tiny binary classifier,
- a deterministic confirmation state machine,
- append-only local logging,
- offline training tools,
- a strict labeling workflow,
- a small CLI.

That is the simplest version that still has a realistic path to being correct.

---

## 14) Implementation Summary

The project should be built as a conservative event detector, not a broad classifier. The model is important, but the data and the event logic are more important. The runtime should remain compact. The training pipeline should remain strict. The labeling workflow should remain honest about uncertainty. The result should be reliable enough to count full snorts in practice without becoming brittle or over-engineered.
