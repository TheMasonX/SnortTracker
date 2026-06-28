"""
Shared configuration for the SnortTracker runtime.

All tunable parameters live here so training and inference
can be kept consistent.  Update this file deliberately and
version it alongside model exports.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio capture and preprocessing settings."""

    sample_rate: int = 16000          # Hz
    channels: int = 1                 # mono
    window_size_ms: int = 25          # ms per analysis window
    hop_size_ms: int = 10             # ms stride between windows
    dtype: str = "float32"            # internal numeric type

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_size_ms / 1000)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_size_ms / 1000)


@dataclass
class GateConfig:
    """Cheap prefilter thresholds.  Keep conservative."""

    rms_energy_threshold: float = 0.01      # min RMS to pass gate
    high_freq_energy_ratio: float = 0.15     # min ratio of HF energy
    zero_crossing_rate_min: float = 0.02
    zero_crossing_rate_max: float = 0.40
    min_event_duration_ms: int = 80          # shortest plausible snort
    max_event_duration_ms: int = 600         # longest plausible snort


@dataclass
class FeatureConfig:
    """Feature extraction settings — must match training exactly."""

    n_mels: int = 40
    n_fft: int = 512
    hop_length: int = 160          # 10 ms at 16 kHz
    f_min: int = 80
    f_max: int = 8000
    normalize: bool = True


@dataclass
class InferenceConfig:
    """Model inference settings."""

    model_path: str = "models/snort-cnn-v1.onnx"
    probability_threshold: float = 0.70  # min confidence to treat as candidate
    backend: str = "onnxruntime"         # inference runtime


@dataclass
class StateMachineConfig:
    """Event confirmation state machine timings."""

    confirmation_windows: int = 3        # consecutive positive windows to confirm
    cooldown_seconds: float = 1.5        # refractory period after a count
    candidate_timeout_seconds: float = 0.8  # max time to stay in candidate state
    min_gate_agreement: int = 2          # gate must agree on at least N windows


@dataclass
class LogConfig:
    """Local logging settings."""

    log_dir: str = "logs"
    log_filename: str = "snort_events.log"
    count_filename: str = "count.txt"
    append_only: bool = True
    utc_timestamps: bool = True


@dataclass
class RuntimeConfig:
    """Top-level configuration aggregating all subsystems."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # Convenience helpers --------------------------------------------------
    @property
    def log_file_path(self) -> Path:
        return Path(self.log.log_dir) / self.log.log_filename

    @property
    def count_file_path(self) -> Path:
        return Path(self.log.log_dir) / self.log.count_filename


# ---------------------------------------------------------------------------
# Default instance — import this directly or override fields as needed
# ---------------------------------------------------------------------------
config = RuntimeConfig()
