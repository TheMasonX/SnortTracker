# Snort Counter System — MVP Implementation Plan
## Detailed Step-by-Step Build Report

**Purpose:** This document lays out a practical implementation plan for the MVP only. It focuses on a simple, reliable, portable snort counter that runs locally, counts full snorts, and avoids overbuilding.

**Design principle:** keep the runtime small, deterministic, testable, and easy to port.

---

## 1. MVP Definition

### 1.1 What the MVP must do
The MVP must:

1. Continuously listen through a USB microphone.
2. Detect candidate snort-like audio using cheap signal gating.
3. Run a tiny binary classifier on only promising windows.
4. Confirm a full snort using a small state machine with a short confirmation window and cooldown.
5. Increment a local count when a snort is confirmed.
6. Append each detected event to a local log file.
7. Provide a minimal CLI for viewing, resetting, and tailing counts/logs.
8. Support offline training on a desktop machine and deployment on a Raspberry Pi-class device.

### 1.2 What the MVP must not do
The MVP must not include:

- dashboards
- cloud sync
- mobile apps
- multi-class audio classification
- person identification or biometric matching
- emotional context detection
- partial-snort counting
- complex analytics
- remote services

### 1.3 Target quality bar
The MVP is successful if it is:

- directionally accurate for the intended user
- stable during long listening sessions
- easy to inspect and debug
- portable across development machines and the target device
- simple enough that future changes do not break the core loop

---

## 2. System Shape

### 2.1 Runtime pipeline
The MVP runtime should follow this sequence:

1. audio capture
2. cheap gating
3. model inference
4. event confirmation state machine
5. counter update
6. append-only logging
7. CLI inspection

### 2.2 Recommended packaging
Keep the runtime logically separated, but do not over-fragment the codebase. A practical structure is:

```text
snort-counter/
├── runtime/
│   ├── capture.py
│   ├── gating.py
│   ├── inference.py
│   ├── state_machine.py
│   ├── logging.py
│   └── config.py
├── dataset/
│   ├── collect.py
│   ├── slice.py
│   ├── label.py
│   ├── augment.py
│   └── preprocess.py
├── train/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── cli/
│   └── main.py
├── models/
├── logs/
├── tests/
└── README.md
```

> **Note:** The `Data/` directory is reserved for MemorySmith wiki deployment.
> All Python development uses a virtual environment (`.venv/`) for portability.

This keeps runtime, training, and test concerns separate without creating unnecessary complexity.

---

## 3. Detailed Implementation Plan

## Phase 0 — Lock the MVP requirements

### Goal
Remove ambiguity before writing code.

### Steps
1. Write a one-page requirements note.
2. Define exactly what qualifies as a full snort.
3. Define what does not qualify:
   - partial snort
   - sniff
   - cough
   - laugh
   - breath burst
   - throat clear
   - ambiguous nasal burst
4. Decide the target deployment baseline:
   - original Raspberry Pi Zero as the conservative baseline
   - treat Pi Zero 2 W as a better-performing variant, not the baseline
5. Choose the initial audio format:
   - 16 kHz
   - mono
   - fixed window length
6. Decide the MVP log schema.
7. Decide the acceptance metrics for the first pass:
   - event-level recall on confirmed full snorts
   - false positives on hard negative audio
   - runtime stability over long sessions

### Output
A short requirements file that becomes the source of truth for the rest of the project.

### Exit criteria
All ambiguous terms are defined clearly enough that two different developers would make the same implementation choice.

---

## Phase 1 — Build the repository skeleton

### Goal
Create a clean structure for code, data, and tests.

### Steps
1. Create the repository root.
2. Add the runtime, training, data, cli, models, logs, and tests directories.
3. Add a `.gitignore` that excludes:
   - raw recordings
   - processed audio
   - generated features
   - model checkpoints
   - local logs
   - build artifacts
4. Add a minimal `README.md` with:
   - purpose
   - architecture summary
   - setup steps
   - how to run tests
   - how to run the CLI
5. Create a shared config file for:
   - sample rate
   - window size
   - gating thresholds
   - probability threshold
   - confirmation window length
   - cooldown length
   - log file path
6. Add a small example configuration with safe defaults.
7. Create a placeholder test suite with one smoke test.

### Output
A repository that can be cloned and understood quickly.

### Exit criteria
The project structure is stable enough that runtime and training code can be added without reorganizing the repository.

---

## Phase 2 — Define the audio contract

### Goal
Make audio assumptions explicit and consistent between training and inference.

### Steps
1. Choose one audio sampling rate for everything.
2. Normalize all audio to mono.
3. Decide the live capture window size.
4. Decide whether inference uses:
   - the same window size as capture, or
   - a smaller feature window derived from the capture buffer
5. Define a single pre-processing pipeline used by both training and runtime.
6. Decide how silence and underflow are handled.
7. Decide how clipped or malformed recordings are handled.
8. Define the expected waveform dtype and numeric range.
9. Write a small utility that validates input audio files against the contract.

### Recommendation
Use a single shared preprocessing path so the training pipeline and the runtime cannot drift apart.

### Output
A documented audio contract and a validation utility.

### Exit criteria
Any WAV file that enters the pipeline can be checked for compatibility before it reaches training or inference.

---

## Phase 3 — Implement audio capture

### Goal
Continuously collect audio with minimal overhead and no dropped-window surprises.

### Steps
1. Implement a capture component that opens the USB microphone.
2. Read audio continuously in the background.
3. Store samples in a ring buffer.
4. Expose a method that returns the next fixed window of audio.
5. Attach metadata to each window:
   - timestamp
   - buffer health
   - underflow or overflow flag
   - capture latency if available
6. Add graceful handling for:
   - device not found
   - device disconnect
   - microphone read failure
   - empty buffer
7. Add a dry-run mode that reads from a WAV file instead of live hardware.
8. Write a capture smoke test using a sample recording.
9. Record dropped-window counts so the runtime can be inspected later.

### Implementation notes
- Keep the capture API stateful.
- Prefer boring, explicit code over clever abstractions.
- The capture layer should never silently lose data.

### Output
A reliable stream source for the rest of the system.

### Exit criteria
The system can run for a meaningful period and continue returning fixed-size windows without crashing.

---

## Phase 4 — Implement cheap gating

### Goal
Reject obvious non-events before neural inference is called.

### Steps
1. Define the gate inputs:
   - RMS energy
   - high-frequency energy
   - zero-crossing rate
   - event duration estimate
2. Implement each metric as a simple helper.
3. Combine metrics into a conservative candidate filter.
4. Make gating thresholds configurable.
5. Keep the gate intentionally permissive enough to avoid missing real snorts.
6. Log why a window passed or failed the gate.
7. Add a debug mode that emits gate metrics for inspection.
8. Write tests for:
   - silence
   - speech
   - laughter
   - coughing
   - breathing
   - obvious snort-like bursts
9. Measure gate pass rate on a small labeled dataset.

### Design rule
Gating is only a cost-saving filter. It is not the final decision maker.

### Output
A cheap first-pass filter with measurable behavior.

### Exit criteria
The gate rejects obvious negatives while still passing enough candidate snorts for the classifier to evaluate.

---

## Phase 5 — Build the dataset pipeline

### Goal
Create a trustworthy training dataset with a strict label policy.

### Steps
1. Create a folder structure for:
   - gold positives
   - silver positives
   - negatives
   - ignored or uncertain clips
2. Define a labeling guide.
3. Define what makes a gold positive:
   - clearly full snort
   - human confirmed
   - no ambiguity
4. Define what belongs in the ignore bucket:
   - partial events
   - borderline nasal bursts
   - uncertain clips
5. Create a slicing utility to cut long recordings into windows.
6. Create a mining utility that finds candidate positives from long sessions using the gate.
7. Create a review workflow for candidate clips.
8. Create a utility to sample hard negatives from non-snort segments.
9. Add preprocessing for:
   - normalization
   - trimming or padding
   - feature extraction
   - optional augmentation
10. Add a dataset manifest format so each clip can be traced back to its source.

### Recommended label policy
Do not force ambiguous clips into the negative set. Excluding uncertain examples is safer than poisoning the boundary.

### Output
A clean dataset pipeline with traceable labels and source provenance.

### Exit criteria
A small but reliable gold dataset can be assembled and reproduced from the same raw recordings.

---

## Phase 6 — Assemble the first training set

### Goal
Prepare a balanced first-pass dataset for model development.

### Steps
1. Collect a first set of clearly labeled positives.
2. Collect hard negatives from realistic non-snort sounds.
3. Review any mined candidate positives manually.
4. Keep downloaded audio only as a supplement, not as the truth source.
5. Remove duplicate or low-quality clips.
6. Split the data into train, validation, and test sets.
7. Ensure the test set is held out before any threshold tuning.
8. Make sure the validation and test sets include hard negatives.
9. Verify that no raw recording leaks across splits.
10. Freeze the dataset version before training.

### Recommended data policy
The intended user’s own confirmed snorts should carry the most weight. Other sources are supporting material, not the center of truth.

### Output
A frozen dataset version ready for training.

### Exit criteria
The dataset is stable, auditable, and split in a way that supports honest evaluation.

---

## Phase 7 — Train the tiny model

### Goal
Train a compact binary classifier that works with the shared audio contract.

### Steps
1. Define a tiny CNN architecture.
2. Keep the model small enough to be practical on low-power hardware.
3. Use binary output with a sigmoid.
4. Train on the preprocessed training set.
5. Weight the intended user’s positive examples higher than generic positives.
6. Use early stopping or similar safeguards.
7. Measure performance on the validation set after each epoch.
8. Inspect false positives and false negatives manually.
9. Adjust augmentation, thresholding, or class weighting if needed.
10. Select the best checkpoint based on event-level validation behavior, not just frame accuracy.

### Key rule
Do not optimize for abstract accuracy alone. Optimize for the counting task.

### Output
A trained checkpoint that is good enough to export and test in runtime.

### Exit criteria
The model produces stable snort probabilities on held-out examples and does not obviously overfit the training set.

---

## Phase 8 — Export the model for deployment

### Goal
Turn the training artifact into something the runtime can use on the target device.

### Steps
1. Choose one deployment path.
2. Export the trained model into the selected portable format.
3. If quantization is used, verify that the exported model still behaves acceptably.
4. Store model metadata:
   - training date
   - dataset version
   - preprocessing version
   - threshold version
   - export format
5. Test the exported model on desktop inference first.
6. Confirm that the runtime preprocessing matches the training preprocessing exactly.
7. Save the final model in the models directory with a versioned name.

### Recommended deployment choice
Pick one runtime path and standardize on it rather than supporting several weakly.

### Output
A versioned deployment artifact and a matching metadata record.

### Exit criteria
The exported model can be loaded and run outside the training code without preprocessing drift.

---

## Phase 9 — Implement the event confirmation state machine

### Goal
Count full snorts once, not repeatedly.

### Steps
1. Define the states:
   - idle
   - candidate
   - confirming
   - counted
   - cooldown
2. Decide the transition rules for each state.
3. Require multiple consecutive positive windows before counting.
4. Require the gate to agree with the classifier.
5. Add a refractory period after a counted event.
6. Reset confirmation state after a timeout.
7. Prevent multiple counts from one long event.
8. Prevent a single event from being split into multiple counts.
9. Write unit tests for state transitions.
10. Write integration tests that replay audio clips through the state machine.

### Design rule
The state machine is the correctness layer. It should be deterministic and easy to reason about.

### Output
A tested event counter that turns model outputs into actual snort events.

### Exit criteria
One real snort is counted once, and obvious double-counting cases are blocked.

---

## Phase 10 — Implement local logging and counting

### Goal
Persist event history locally in a simple, robust format.

### Steps
1. Decide the log schema.
2. Include at least:
   - ISO 8601 UTC timestamp
   - confidence score
   - event id
   - model version
   - config version
3. Append events to a local text file.
4. Make the writes atomic enough for normal power loss scenarios.
5. Keep a separate count file only if needed.
6. Make reset behavior explicit and safe.
7. Add a log tail command.
8. Add a log parse utility for tests.
9. Verify that logs survive a restart.
10. Verify that log lines remain easy to parse and diff.

### Recommended format
Use append-only plain text for the MVP. It is easier to inspect and test than a database.

### Output
A local, durable record of each confirmed snort event.

### Exit criteria
The count can be reconstructed from logs, and the logs are readable without extra tooling.

---

## Phase 11 — Build the CLI

### Goal
Expose the minimum user-facing commands needed to operate the MVP.

### Steps
1. Implement a `view` command to show the current count.
2. Implement a `reset` command to clear count and logs.
3. Implement a `tail` command to show recent events.
4. Implement a `status` command to show:
   - model loaded or not
   - capture health
   - gate pass rate
   - dropped-window count
   - last event time
5. Keep the CLI text-only.
6. Make the CLI work on the development machine and the target device.
7. Add help text and examples.
8. Add tests for command behavior.
9. Ensure the CLI does not require the runtime to be listening continuously.
10. Make commands fail clearly if files are missing or corrupted.

### Output
A minimal but useful command-line interface.

### Exit criteria
A non-technical user can inspect the count and reset the system without touching code.

---

## Phase 12 — Integrate the runtime loop

### Goal
Connect capture, gate, inference, state machine, and logging into one loop.

### Steps
1. Wire capture output into the gate.
2. Pass gated windows into the model.
3. Feed model probabilities and gate results into the state machine.
4. Trigger logging only on confirmed events.
5. Add runtime counters for:
   - processed windows
   - gate passes
   - model calls
   - confirmed snorts
   - dropped windows
6. Add defensive handling for empty outputs or inference failures.
7. Add a dry-run mode using stored audio files.
8. Add a live mode using the USB microphone.
9. Make sure the loop is stoppable cleanly.
10. Add console status output for debugging.

### Output
A working end-to-end MVP pipeline.

### Exit criteria
The system can listen continuously, count events, and write logs without manual intervention.

---

## Phase 13 — Add test coverage

### Goal
Prove that the system behaves correctly in the important cases.

### Steps
1. Create unit tests for:
   - gating metrics
   - state machine transitions
   - logging format
   - configuration parsing
2. Create offline integration tests using stored WAV fixtures.
3. Test:
   - one snort count per true event
   - no count on silence
   - no count on speech
   - no count on laughter
   - no double-count from one long event
   - cooldown behavior
   - reset behavior
4. Add a hardware smoke test that runs the live loop for a short period.
5. Add a regression fixture for each bug found during development.
6. Make test failures readable.
7. Include a small test dataset in the repo or in a known local fixture path.
8. Run tests on every meaningful code change.

### Output
A test suite that protects the core behavior.

### Exit criteria
The most important failure modes are covered by automated tests.

---

## Phase 14 — Run calibration and threshold tuning

### Goal
Adjust the system to the real environment without changing the core design.

### Steps
1. Record a short calibration session in the target room.
2. Measure ambient noise level.
3. Measure gate metrics on silence and normal room audio.
4. Tune gating thresholds conservatively.
5. Tune the probability threshold.
6. Tune the confirmation window length.
7. Tune the cooldown period.
8. Verify the settings on fresh audio.
9. Save calibration defaults to config.
10. Re-run the test suite after tuning.

### Recommended approach
Tune on a small calibration set, then freeze the settings and evaluate again on held-out data.

### Output
A calibrated set of defaults for the target environment.

### Exit criteria
The system works reasonably in the real room without becoming overly sensitive.

---

## 4. Suggested MVP Deliverables

The MVP should end with these concrete deliverables:

1. repository skeleton
2. shared config file
3. audio capture module
4. cheap gating module
5. preprocessing pipeline
6. dataset collection and slicing tools
7. labeling guide
8. initial curated dataset
9. tiny trained binary model
10. deployment export artifact
11. confirmation state machine
12. local append-only logging
13. CLI commands for view, reset, tail, and status
14. unit and integration tests
15. calibration notes and default thresholds

---

## 5. Acceptance Criteria

The MVP is ready for first use when all of the following are true:

1. The system can run continuously for a meaningful session.
2. The pipeline remains stable on the target device.
3. Full snorts are detected as events rather than as raw frame hits.
4. A single snort is counted once.
5. Silence and ordinary negative sounds do not generate obvious false counts.
6. Logs are local, readable, and reconstructable.
7. The CLI works without requiring a dashboard.
8. The model and preprocessing contract are consistent between training and runtime.
9. Tests cover the core failure modes.
10. Ambiguous data is excluded from the gold training set.

---

## 6. Common Failure Modes to Watch For

### 6.1 Bad labels
If ambiguous clips are forced into the negative class, the classifier will learn a distorted boundary.

### 6.2 Overly aggressive gating
If the gate is too strict, real snorts will never reach inference.

### 6.3 Drift between training and runtime preprocessing
If the feature extraction differs, the model will fail even if it trained well.

### 6.4 Missing cooldown
Without a cooldown, one event may be counted multiple times.

### 6.5 Too many modules too early
If the codebase is split too finely before behavior is stable, debugging becomes harder rather than easier.

### 6.6 Trusting auto-mined labels too much
Auto-mined positives should be reviewed, not assumed correct.

---

## 7. Practical Build Order

If the work needs to be done in the fewest risky steps, the recommended order is:

1. define the label policy
2. create the repository skeleton
3. implement audio capture
4. implement gating
5. implement offline slicing and preprocessing
6. collect and curate the first dataset
7. train the first tiny model
8. export the model
9. implement the state machine
10. implement logging
11. wire the runtime loop
12. add CLI commands
13. add tests
14. calibrate thresholds
15. run hardware smoke tests
16. freeze the MVP release

This order minimizes rework by solving data and event definition before polishing the runtime.

---

## 8. Final Recommendation

The safest MVP is not the most feature-rich one. It is the one that is easiest to understand, easiest to test, and hardest to miscount. Keep the runtime small, keep the dataset clean, and make the state machine do the counting. Everything else is secondary.

---

## 9. Appendix — Minimal first-pass file list

A practical first-pass implementation could start with these files:

```text
runtime/
├── capture.py
├── gating.py
├── inference.py
├── state_machine.py
├── logging.py
└── config.py

dataset/
├── collect.py
├── slice.py
├── label.py
├── preprocess.py
└── augment.py

train/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
└── export.py

cli/
└── main.py

tests/
├── test_gating.py
├── test_state_machine.py
├── test_logging.py
├── test_capture.py
└── test_runtime_smoke.py
```

