"""
Tests for the dataset collection tooling.

Covers: manifest CRUD, CSV roundtrip, window slicing,
label annotation, session splitting, preprocessing.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from runtime.audio_contract import DTYPE, SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES
from dataset.manifest import Manifest, ManifestEntry, LabelStatus, DataSplit
from dataset.slicer import Slicer, SlicedWindow
from dataset.labeler import (
    LabelPolicy,
    SessionSplitter,
    validate_manifest_labels,
)
from dataset.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_seconds: float = 1.0, freq: float = 440.0) -> Path:
    """Write a mono 16-bit PCM WAV at 16kHz."""
    n_samples = int(SAMPLE_RATE * duration_seconds)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    audio = (0.1 * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)
    i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(i16.tobytes())
    return path


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_empty_manifest(self):
        m = Manifest(Path("/tmp/nonexistent.csv"))
        assert len(m) == 0
        assert m.label_counts == {}

    def test_add_and_get(self):
        m = Manifest(Path("/tmp/test.csv"))
        entry = m.add(
            session_id="S001",
            path=Path("recordings/s001.wav"),
            duration_seconds=30.0,
        )
        assert entry.session_id == "S001"
        assert m.get("S001") is entry
        assert len(m) == 1

    def test_label_assignment(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="S001", path=Path("s001.wav"), duration_seconds=10.0)
        m.label("S001", LabelStatus.POSITIVE, notes="confirmed snort at 2.1s")
        entry = m.get("S001")
        assert entry.label_status == LabelStatus.POSITIVE
        assert "snort" in entry.notes

    def test_split_assignment(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="S001", path=Path("s001.wav"), duration_seconds=10.0)
        m.assign_split("S001", DataSplit.TRAIN)
        assert m.get("S001").split == DataSplit.TRAIN

    def test_by_label(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="S001", path=Path("a.wav"), duration_seconds=5.0,
              label_status=LabelStatus.POSITIVE)
        m.add(session_id="S002", path=Path("b.wav"), duration_seconds=5.0,
              label_status=LabelStatus.NEGATIVE)
        m.add(session_id="S003", path=Path("c.wav"), duration_seconds=5.0,
              label_status=LabelStatus.IGNORE)

        assert len(m.by_label(LabelStatus.POSITIVE)) == 1
        assert len(m.by_label(LabelStatus.NEGATIVE)) == 1
        assert len(m.labeled()) == 2

    def test_label_counts(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="S001", path=Path("a.wav"), duration_seconds=5.0,
              label_status=LabelStatus.POSITIVE)
        m.add(session_id="S002", path=Path("b.wav"), duration_seconds=5.0,
              label_status=LabelStatus.POSITIVE)
        m.add(session_id="S003", path=Path("c.wav"), duration_seconds=5.0,
              label_status=LabelStatus.NEGATIVE)
        counts = m.label_counts
        assert counts["positive"] == 2
        assert counts["negative"] == 1

    def test_remove(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="S001", path=Path("a.wav"), duration_seconds=5.0)
        m.remove("S001")
        assert m.get("S001") is None
        assert len(m) == 0

    def test_label_missing_session_raises(self):
        m = Manifest(Path("/tmp/test.csv"))
        with pytest.raises(KeyError):
            m.label("NOEXIST", LabelStatus.POSITIVE)

    def test_iter(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="A", path=Path("a.wav"), duration_seconds=1.0)
        m.add(session_id="B", path=Path("b.wav"), duration_seconds=2.0)
        ids = sorted(e.session_id for e in m)
        assert ids == ["A", "B"]


class TestManifestCSV:
    def test_roundtrip(self, tmp_path: Path):
        csv_path = tmp_path / "manifest.csv"
        m1 = Manifest(csv_path)
        m1.add(session_id="S001", path=Path("rec/s001.wav"), duration_seconds=10.5,
               label_status=LabelStatus.POSITIVE, split=DataSplit.TRAIN,
               notes="confirmed")
        m1.add(session_id="S002", path=Path("rec/s002.wav"), duration_seconds=15.0,
               label_status=LabelStatus.NEGATIVE, split=DataSplit.VAL)
        m1.save()

        m2 = Manifest(csv_path)
        assert len(m2) == 2
        e = m2.get("S001")
        assert e.label_status == LabelStatus.POSITIVE
        assert e.split == DataSplit.TRAIN
        assert e.duration_seconds == pytest.approx(10.5)
        assert e.notes == "confirmed"

    def test_load_nonexistent(self):
        m = Manifest(Path("/tmp/definitely_not_real.csv"))
        assert len(m) == 0


# ---------------------------------------------------------------------------
# Slicer
# ---------------------------------------------------------------------------


class TestSlicer:
    def test_slice_exact_multiple(self):
        """Audio exactly N windows long (in hop-aligned terms)."""
        n_windows = 4
        # n_windows * HOP_SAMPLES produces ceil(n_windows*HOP/HOP) = n_windows windows
        audio = np.random.randn(HOP_SAMPLES * n_windows).astype(DTYPE)
        slicer = Slicer()
        windows = slicer.slice(audio, session_id="test")
        assert len(windows) == n_windows
        for i, w in enumerate(windows):
            assert w.window_index == i
            assert w.audio.shape == (WINDOW_SAMPLES,)

    def test_slice_with_overlap(self):
        """Overlapping windows (hop < window)."""
        # 1 full window + 1 hop → 2 windows (starts at 0 and hop)
        total = WINDOW_SAMPLES + HOP_SAMPLES
        audio = np.random.randn(total).astype(DTYPE)
        slicer = Slicer()
        windows = slicer.slice(audio, session_id="test")
        # ceil((WINDOW_SAMPLES + HOP_SAMPLES) / HOP_SAMPLES) = ceil(560/160) = 4
        expected = (total + HOP_SAMPLES - 1) // HOP_SAMPLES
        assert len(windows) == expected

    def test_slice_short_audio(self):
        """Audio shorter than one window produces one zero-padded window."""
        audio = np.random.randn(100).astype(DTYPE)
        slicer = Slicer()
        windows = slicer.slice(audio, session_id="test")
        assert len(windows) == 1
        assert windows[0].audio.shape == (WINDOW_SAMPLES,)

    def test_count_windows(self):
        slicer = Slicer()
        # 1 second at 16kHz → 16000 samples
        # ceil(16000 / 160) = 100
        n_samples = int(1.0 * SAMPLE_RATE)
        expected = (n_samples + HOP_SAMPLES - 1) // HOP_SAMPLES
        assert slicer.count_windows(1.0) == expected

    def test_slice_label_applied(self):
        audio = np.random.randn(WINDOW_SAMPLES).astype(DTYPE)
        slicer = Slicer()
        windows = slicer.slice(audio, session_id="S1", label=LabelStatus.NEGATIVE)
        assert windows[0].label == LabelStatus.NEGATIVE

    def test_slice_with_annotations(self):
        """Per-range labels with positive overriding negative."""
        audio = np.random.randn(WINDOW_SAMPLES * 4).astype(DTYPE)
        slicer = Slicer()
        # First 50ms positive, everything else negative
        windows = slicer.slice_with_annotations(
            audio,
            session_id="S1",
            positive_ranges=[(0.0, 0.05)],
            negative_ranges=[(0.0, 0.2)],
            default_label=LabelStatus.IGNORE,
        )
        assert len(windows) > 0
        # First window starts at 0, covers 0-25ms → positive
        assert windows[0].label == LabelStatus.POSITIVE

    def test_slice_file(self, tmp_path: Path):
        wav_path = tmp_path / "slice_test.wav"
        _make_wav(wav_path, duration_seconds=0.5)
        slicer = Slicer()
        windows = slicer.slice_file(wav_path, label=LabelStatus.UNLABELED)
        assert len(windows) == slicer.count_windows(0.5)
        for w in windows:
            assert w.audio.shape == (WINDOW_SAMPLES,)
            assert w.audio.dtype == DTYPE


# ---------------------------------------------------------------------------
# Label policy & session splitter
# ---------------------------------------------------------------------------


class TestSessionSplitter:
    def test_default_ratios(self):
        splitter = SessionSplitter()
        assert splitter.train_ratio + splitter.val_ratio + splitter.test_ratio == pytest.approx(1.0)

    def test_rejects_invalid_ratios(self):
        with pytest.raises(ValueError):
            SessionSplitter(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_assigns_all_splits(self):
        m = Manifest(Path("/tmp/test.csv"))
        for i in range(10):
            m.add(session_id=f"S{i:03d}", path=Path(f"rec/{i}.wav"),
                  duration_seconds=10.0, label_status=LabelStatus.NEGATIVE)

        splitter = SessionSplitter(seed=42)
        splitter.assign_splits(m)

        train = m.by_split(DataSplit.TRAIN)
        val = m.by_split(DataSplit.VAL)
        test = m.by_split(DataSplit.TEST)

        assert len(train) + len(val) + len(test) == 10
        assert len(train) == 7  # 70% of 10
        assert len(val) == 1    # 15% (floor)
        assert len(test) == 2   # remainder

    def test_reproducible(self):
        m1 = Manifest(Path("/tmp/a.csv"))
        m2 = Manifest(Path("/tmp/b.csv"))
        for i in range(20):
            m1.add(session_id=f"S{i:03d}", path=Path(f"{i}.wav"), duration_seconds=5.0)
            m2.add(session_id=f"S{i:03d}", path=Path(f"{i}.wav"), duration_seconds=5.0)

        SessionSplitter(seed=42).assign_splits(m1)
        SessionSplitter(seed=42).assign_splits(m2)

        for e1, e2 in zip(sorted(m1.entries, key=lambda x: x.session_id),
                          sorted(m2.entries, key=lambda x: x.session_id)):
            assert e1.split == e2.split, f"Mismatch for {e1.session_id}"


class TestValidateManifest:
    def test_empty_warns(self):
        m = Manifest(Path("/tmp/test.csv"))
        warnings = validate_manifest_labels(m)
        assert len(warnings) == 2  # no positives, no negatives

    def test_labeled_without_split_warns(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="S001", path=Path("a.wav"), duration_seconds=5.0,
              label_status=LabelStatus.POSITIVE)
        warnings = validate_manifest_labels(m)
        assert any("no data split" in w.lower() for w in warnings)

    def test_complete_manifest_no_warnings(self):
        m = Manifest(Path("/tmp/test.csv"))
        m.add(session_id="S001", path=Path("a.wav"), duration_seconds=5.0,
              label_status=LabelStatus.POSITIVE, split=DataSplit.TRAIN)
        m.add(session_id="S002", path=Path("b.wav"), duration_seconds=5.0,
              label_status=LabelStatus.NEGATIVE, split=DataSplit.TRAIN)
        warnings = validate_manifest_labels(m)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    def test_process_windows(self):
        """Process a list of sliced windows into features + labels."""
        # Use exact hop-aligned count: HOP_SAMPLES * 3 produces 3 windows
        n_windows = 3
        audio = np.random.randn(HOP_SAMPLES * n_windows).astype(DTYPE)
        slicer = Slicer()
        windows = slicer.slice(audio, session_id="test", label=LabelStatus.POSITIVE)

        prep = Preprocessor()
        features, labels, kept = prep.process_windows(windows)

        assert features.shape == (n_windows, 40)
        assert features.dtype == DTYPE
        assert labels.shape == (n_windows,)
        assert np.all(labels == 1)
        assert len(kept) == n_windows

    def test_excludes_ignore_by_default(self):
        audio = np.random.randn(WINDOW_SAMPLES * 2).astype(DTYPE)
        slicer = Slicer()
        windows = [
            SlicedWindow(
                audio=audio[:WINDOW_SAMPLES],
                label=LabelStatus.POSITIVE,
                session_id="S1", window_index=0,
                start_sample=0, start_seconds=0.0,
            ),
            SlicedWindow(
                audio=audio[WINDOW_SAMPLES:],
                label=LabelStatus.IGNORE,
                session_id="S1", window_index=1,
                start_sample=WINDOW_SAMPLES,
                start_seconds=WINDOW_SAMPLES / SAMPLE_RATE,
            ),
        ]

        prep = Preprocessor()
        features, labels, kept = prep.process_windows(windows)

        assert features.shape == (1, 40)
        assert labels[0] == 1

    def test_process_empty(self):
        prep = Preprocessor()
        features, labels, kept = prep.process_windows([])
        assert features.shape == (0, 40)
        assert labels.shape == (0,)

    def test_extractor_consistency(self):
        """Same audio through slicer → preprocessor should match direct extraction."""
        # Use HOP_SAMPLES to guarantee exactly 1 window
        audio = np.random.randn(HOP_SAMPLES).astype(DTYPE)

        from runtime.features import FeatureExtractor
        direct = FeatureExtractor().extract(audio)

        slicer = Slicer()
        windows = slicer.slice(audio, session_id="test", label=LabelStatus.POSITIVE)
        assert len(windows) == 1, f"Expected 1 window, got {len(windows)}"

        prep = Preprocessor()
        features, _, _ = prep.process_windows(windows)

        assert np.allclose(features[0], direct, rtol=1e-5)
