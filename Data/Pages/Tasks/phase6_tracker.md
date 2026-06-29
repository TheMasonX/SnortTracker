# Phase 6 — End-to-End Integration v0.1 — Task Tracker

**Status:** ✅ Complete  
**Started:** 2026-06-28  
**Completed:** 2026-06-28  
**Goal:** Wire everything together and ship a working v0.1 prototype.

---

## Task List

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Create `runtime/classifier.py` — GateHeuristicClassifier | ✅ | `Classifier` ABC + `GateHeuristicClassifier` + `create_classifier()` factory |
| 2 | Refactor `cli/main.py` to use classifier module | ✅ | Removed hardcoded 0.85/0.10; uses `create_classifier()` factory |
| 3 | Write `tests/test_integration.py` — real pipeline test | ✅ | 14 tests: classifier, full pipeline, config consistency |
| 4 | Add `## Privacy` section to README.md | ✅ | Data map, purge instructions, local-first guarantees |
| 5 | Add `purge` CLI command | ✅ | Confirmation prompt, `-y` flag, wipes logs + rotated archives |
| 6 | Run full test suite | ✅ | 161/161 pass |
| 7 | Update ROADMAP.md | ✅ | Phase 6 marked complete |

---

## Exit Criteria

- [x] `python -m cli.main start --input tests/fixtures/test_snorts.wav` works end-to-end with classifier module
- [x] Placeholder classifier is a proper injectable module (swap ready for Phase 10)
- [x] Integration test exercises the full pipeline with a WAV fixture
- [x] `## Privacy` section in README documents all stored data + purge procedure
- [x] `purge` CLI command with confirmation
- [x] Integration tests replace the old skeleton-only smoke approach

---

## Notes

- `runtime/classifier.py` defines `Classifier` ABC with `predict(audio, gate_result) -> float`
- `GateHeuristicClassifier` maps gate pass + RMS/HF ratio to probability [0,1]
- CLI now prints model name + version on `start`
- `purge` with no `-y` prompts interactively; `purge -y` skips for automation
