"""
Smoke test for the runtime skeleton.

Verifies that the project structure is in place and that the
config module loads correctly.  This test should always pass
after Phase 1 is complete.
"""

import importlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Repository structure
# ---------------------------------------------------------------------------
EXPECTED_DIRS = [
    "runtime",
    "dataset",
    "train",
    "cli",
    "models",
    "logs",
    "tests",
]

EXPECTED_RUNTIME_FILES = [
    "config.py",
    "__init__.py",
]


def test_repo_structure_exists():
    """All expected top-level directories exist."""
    root = Path(__file__).resolve().parent.parent
    for name in EXPECTED_DIRS:
        dir_path = root / name
        assert dir_path.is_dir(), f"Missing directory: {name}"


def test_config_module_loads():
    """The shared config module can be imported and has expected fields."""
    config_mod = importlib.import_module("runtime.config")

    assert hasattr(config_mod, "config"), "Expected top-level `config` instance"
    cfg = config_mod.config

    # Audio
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.channels == 1

    # Gate
    assert cfg.gate.rms_energy_threshold > 0

    # Features
    assert cfg.features.n_mels == 40

    # Inference
    assert 0 < cfg.inference.probability_threshold <= 1.0

    # State machine
    assert cfg.state_machine.confirmation_windows >= 1
    assert cfg.state_machine.cooldown_seconds > 0

    # Log
    assert cfg.log.log_filename == "snort_events.log"


def test_config_derived_properties():
    """Derived properties return sensible values."""
    cfg = importlib.import_module("runtime.config").config

    assert cfg.audio.window_samples > 0
    assert cfg.audio.hop_samples > 0
    assert cfg.log_file_path.name == "snort_events.log"
    assert cfg.count_file_path.name == "count.txt"


def test_packages_importable():
    """Every package __init__.py exists and the package is importable."""
    for pkg in ["runtime", "dataset", "train", "cli", "tests"]:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import package: {pkg}"
