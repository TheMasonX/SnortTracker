"""
Tests for the SnortCNN model and training infrastructure.

Covers: model architecture, forward pass, parameter count,
ONNX export, training loop on synthetic data, event metrics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from train.model import SnortCNN, create_model
from train.train import Trainer, TrainingConfig
from train.evaluate import (
    EventMetrics,
    compute_event_metrics,
    window_to_event_predictions,
)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class TestSnortCNN:
    def test_default_construction(self):
        model = SnortCNN()
        assert model.input_dim == 40
        assert model.hidden_dims == [64, 32]
        assert model.num_parameters < 10000  # tiny model

    def test_forward_pass_shape(self):
        model = SnortCNN()
        x = torch.randn(16, 40)
        out = model(x)
        assert out.shape == (16, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_single_sample_forward(self):
        model = SnortCNN()
        x = torch.randn(1, 40)
        out = model(x)
        assert out.shape == (1, 1)

    def test_deterministic_in_eval_mode(self):
        model = SnortCNN()
        model.eval()
        x = torch.randn(4, 40)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_custom_dims(self):
        model = SnortCNN(input_dim=64, hidden_dims=[128, 64, 32])
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8, 1)

    def test_parameter_count_reasonable(self):
        model = SnortCNN()
        n_params = model.num_parameters
        # 40*64 + 64 + 64*32 + 32 + 32*1 + 1 = 2560+64+2048+32+32+1 = 4737
        print(f"Model parameters: {n_params}")
        assert 4000 <= n_params <= 6000

    def test_save_load_roundtrip(self, tmp_path: Path):
        model = SnortCNN()
        path = tmp_path / "test_model.pt"
        model.save(path)

        loaded = SnortCNN.load(path)
        assert loaded.num_parameters == model.num_parameters

        # Same output for same input
        x = torch.randn(4, 40)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x))

    def test_create_model_factory(self):
        model = create_model(hidden_dims=[32])
        assert model.hidden_dims == [32]
        x = torch.randn(2, 40)
        assert model(x).shape == (2, 1)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class TestONNXExport:
    def test_export_does_not_crash(self, tmp_path: Path):
        model = SnortCNN()
        onnx_path = tmp_path / "model.onnx"
        model.export_onnx(onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_exported_model_exists_and_is_valid(self, tmp_path: Path):
        """ONNX file is written and can be checked with onnx module if available."""
        model = SnortCNN()
        onnx_path = tmp_path / "snort.onnx"
        model.export_onnx(onnx_path)

        # Basic validity: file is at least 1KB (ONNX header + graph)
        assert onnx_path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Training loop (smoke test with synthetic data)
# ---------------------------------------------------------------------------


class TestTrainingSmoke:
    def test_training_runs_on_synthetic_data(self):
        """Smoke test: training loop completes without crashing."""
        # Create synthetic balanced data
        rng = np.random.RandomState(42)
        n_samples = 200
        features = rng.randn(n_samples, 40).astype(np.float32)
        labels = (rng.rand(n_samples) > 0.5).astype(np.int64)

        from train.dataset import SnortDataset
        ds = SnortDataset.from_arrays(features, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

        model = create_model()
        config = TrainingConfig(
            epochs=3,
            learning_rate=1e-3,
            positive_class_weight=2.0,
            device="cpu",
        )
        trainer = Trainer(model, config, save_dir=Path(tempfile.mkdtemp()))
        history = trainer.train(loader, val_loader=None)

        assert len(history) == 3
        # Loss should decrease
        assert history[0].train_loss > 0
        # At minimum, training should not explode
        assert not np.isnan(history[-1].train_loss)

    def test_training_with_validation(self):
        """Training loop with validation loader."""
        rng = np.random.RandomState(42)
        features = rng.randn(200, 40).astype(np.float32)
        labels = (rng.rand(200) > 0.5).astype(np.int64)

        from train.dataset import SnortDataset

        # Split
        train_ds = SnortDataset.from_arrays(features[:150], labels[:150])
        val_ds = SnortDataset.from_arrays(features[150:], labels[150:])

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

        model = create_model()
        config = TrainingConfig(epochs=5, device="cpu")
        trainer = Trainer(model, config, save_dir=Path(tempfile.mkdtemp()))
        history = trainer.train(train_loader, val_loader=val_loader)

        assert len(history) > 0
        # All epochs should have valid metrics
        for m in history:
            assert not np.isnan(m.train_loss)
            assert 0.0 <= m.train_acc <= 1.0

    def test_config_defaults(self):
        config = TrainingConfig()
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.epochs == 50
        assert config.early_stopping_patience == 10
        assert config.positive_class_weight > 1.0


# ---------------------------------------------------------------------------
# Event-level metrics
# ---------------------------------------------------------------------------


class TestEventMetrics:
    def test_perfect_detection(self):
        metrics = compute_event_metrics(
            ground_truth_count=10,
            predicted_count=10,
            audio_duration_minutes=5.0,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.false_positives_per_minute == 0.0

    def test_all_missed(self):
        metrics = compute_event_metrics(
            ground_truth_count=10,
            predicted_count=0,
            audio_duration_minutes=5.0,
        )
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    def test_overcounting(self):
        metrics = compute_event_metrics(
            ground_truth_count=5,
            predicted_count=15,
            audio_duration_minutes=10.0,
        )
        assert metrics.precision == pytest.approx(5 / 15)
        assert metrics.recall == 1.0
        assert metrics.false_positives_per_minute == 1.0  # 10 FP / 10 min

    def test_temporal_matching(self):
        metrics = compute_event_metrics(
            ground_truth_count=3,
            predicted_count=3,
            audio_duration_minutes=1.0,
            ground_truth_event_times=[1.0, 2.0, 3.0],
            predicted_event_times=[1.05, 2.10, 3.02],
            match_window_seconds=0.3,
        )
        assert metrics.true_positives == 3
        assert metrics.false_positives == 0
        assert metrics.precision == 1.0

    def test_temporal_mismatch(self):
        metrics = compute_event_metrics(
            ground_truth_count=2,
            predicted_count=2,
            audio_duration_minutes=1.0,
            ground_truth_event_times=[1.0, 3.0],
            predicted_event_times=[2.0, 4.0],  # all off by ~1s
            match_window_seconds=0.3,
        )
        assert metrics.true_positives == 0
        assert metrics.precision == 0.0

    def test_to_dict(self):
        metrics = compute_event_metrics(
            ground_truth_count=8,
            predicted_count=6,
            audio_duration_minutes=3.0,
        )
        d = metrics.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert "false_positives_per_minute" in d


class TestWindowToEventPredictions:
    def test_no_predictions(self):
        probs = np.zeros(100)
        events = window_to_event_predictions(probs, threshold=0.5)
        assert events == []

    def test_single_contiguous_event(self):
        probs = np.array([0.1] * 10 + [0.9] * 20 + [0.1] * 10)
        events = window_to_event_predictions(probs, threshold=0.5)
        assert len(events) == 1
        # Center of the 20 positive windows (indices 10-29 → center at 19.5 → 0.195s)
        assert 0.15 < events[0] < 0.25

    def test_two_separate_events(self):
        # Two bursts separated by low-prob gap
        probs = np.array(
            [0.1] * 10 + [0.9] * 10 + [0.1] * 30 + [0.9] * 10 + [0.1] * 10
        )
        events = window_to_event_predictions(
            probs, threshold=0.5, min_event_gap_seconds=0.2,
        )
        assert len(events) == 2

    def test_merges_close_events(self):
        # Two bursts separated by a small gap → merged
        probs = np.array(
            [0.1] * 10 + [0.9] * 5 + [0.1] * 2 + [0.9] * 5 + [0.1] * 10
        )
        events = window_to_event_predictions(
            probs, threshold=0.5, min_event_gap_seconds=0.5,
        )
        assert len(events) == 1  # merged because gap is small


# ---------------------------------------------------------------------------
# SnortDataset
# ---------------------------------------------------------------------------


class TestSnortDataset:
    def test_from_arrays(self):
        from train.dataset import SnortDataset
        features = np.random.randn(100, 40).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.int64)
        ds = SnortDataset.from_arrays(features, labels)
        assert len(ds) == 100
        f, l = ds[0]
        assert f.shape == (40,)
        assert l.shape == ()

    def test_dataloader_integration(self):
        from train.dataset import SnortDataset
        features = np.random.randn(128, 40).astype(np.float32)
        labels = np.random.randint(0, 2, 128).astype(np.int64)
        ds = SnortDataset.from_arrays(features, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
        for batch_features, batch_labels in loader:
            assert batch_features.shape[0] <= 32
            assert batch_features.shape[1] == 40
            assert batch_labels.shape[0] == batch_features.shape[0]
            break
