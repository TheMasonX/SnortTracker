# Phase 8 — Dataset Collection — Task Tracker

**Status:** 🔜 In Progress  
**Started:** 2026-06-28  
**Goal:** Build dataset collection tooling: slicing, labeling, preprocessing, manifest format.

---

## Task List

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Create `dataset/manifest.py` — session/recording manifest | ⏳ | CSV format: session_id, path, duration, labels, split |
| 2 | Create `dataset/slicer.py` — window slicing per contract | ⏳ | Overlapping windows with labels, session-aware splitting |
| 3 | Create `dataset/labeler.py` — label policy management | ⏳ | positive/negative/ignore; gold-standard workflow |
| 4 | Create `dataset/preprocessor.py` — batch feature extraction | ⏳ | Wraps FeatureExtractor; produces (N, n_mels) arrays |
| 5 | Write `tests/test_dataset.py` | ⏳ | Slicing, labeling, preprocessing, manifest roundtrip |
| 6 | Run full test suite | ⏳ | Verify all pass |
| 7 | Update ROADMAP.md + HANDOFF.md | ⏳ | Mark Phase 8 complete |

---

## Exit Criteria

- [ ] Audio files can be sliced into labeled windows consistent with the audio contract
- [ ] Label policy (positive/negative/ignore) is enforced programmatically
- [ ] Session-based train/val/test splitting prevents data leakage
- [ ] Manifest format is round-trippable (write → read → identical)
- [ ] Feature preprocessing pipeline produces (N, 40) arrays ready for training
- [ ] All dataset tests pass

---

## Notes

- Dataset tooling works with any 16kHz mono audio — no real snort recordings needed to test
- Session-based splitting ensures windows from the same recording stay together
- Manifest is plain CSV for human readability and version control
- Label policy: positive = confirmed snort, negative = confirmed non-snort, ignore = ambiguous
- Gold positives require human confirmation — tooling supports this workflow but doesn't automate it
