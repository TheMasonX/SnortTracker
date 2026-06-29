# Phase 9 — Model Training — Task Tracker

**Status:** ✅ Complete  
**Started:** 2026-06-28  
**Completed:** 2026-06-28  
**Goal:** Build a tiny CNN classifier for snort detection, trainable on desktop GPU.

---

## Task List

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Add `torch>=2.0`, `torchaudio>=2.0` to requirements.txt | ✅ | Also added `onnx>=1.16`, `onnxscript>=0.1` |
| 2 | Create `train/model.py` — Tiny CNN architecture | ✅ | FC network: 40→64→32→1, ~4.7k params, ONNX export |
| 3 | Create `train/dataset.py` — PyTorch Dataset class | ✅ | `SnortDataset` wraps Preprocessor + Manifest; DataLoader factory |
| 4 | Create `train/train.py` — training loop | ✅ | Class-weighted BCE, early stopping, checkpointing |
| 5 | Create `train/evaluate.py` — event-level metrics | ✅ | Precision/recall/F1, FP/min, temporal matching, window→event |
| 6 | Write `tests/test_model.py` | ✅ | 23 tests: architecture, forward pass, ONNX export, training, metrics |
| 7 | Run full test suite | ✅ | 215/215 pass |
| 8 | Update ROADMAP.md + HANDOFF.md | ⏳ | Mark Phase 9 complete |

---

## Exit Criteria

- [x] Model forward pass produces (batch_size, 1) output in [0, 1]
- [x] Model has < 50k parameters (~4.7k — fits Pi Zero memory)
- [x] Training loop runs on synthetic data without crashing
- [x] Event-level evaluation computes precision/recall/F1
- [x] `SnortDataset` correctly loads from arrays and manifests
- [x] ONNX export produces valid .onnx file
- [x] 215/215 tests pass

---

## Files Created

| File | Purpose |
|------|---------|
| `train/model.py` | SnortCNN classifier + ONNX export |
| `train/dataset.py` | PyTorch Dataset + DataLoader factories |
| `train/train.py` | Training loop with class weighting, early stopping |
| `train/evaluate.py` | Event-level metrics + window-to-event conversion |
| `train/__init__.py` | Package exports |
| `tests/test_model.py` | 23 tests: architecture, ONNX, training smoke, metrics |

## Notes

- Tiny FC network: 40→64→ReLU→Dropout→32→ReLU→Dropout→1→Sigmoid = 4,737 params
- `Trainer` supports class-weighted BCE to handle imbalanced snort data
- `compute_event_metrics` supports both count-based and temporal-matching evaluation
- `window_to_event_predictions` converts per-window probabilities to event timestamps
- ONNX export uses dynamic batch size for deployment flexibility
- Training requires a manifest CSV (from Phase 8) with labeled sessions
