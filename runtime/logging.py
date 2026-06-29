"""
Append-only local event logging for the SnortTracker runtime.

Writes confirmed snort events to a plaintext log file.  The format
is intentionally simple so logs are human-readable and easy to parse
with standard tools (``grep``, ``awk``, ``tail``).

Log format (CSV-style)::

    ISO8601_UTC, event_id, confidence, model_version, config_version

Rotation is supported via max age (days) and max size (bytes).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runtime.config import LogConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class EventLogger:
    """Append-only event logger with optional rotation."""

    log_path: Path
    max_age_days: int = 30
    max_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    model_version: str = "snort-cnn-v1"
    config_version: str = "1"

    _event_count: int = 0

    def __post_init__(self) -> None:
        # Load existing count from log file
        self._event_count = self._count_lines()

    @classmethod
    def from_config(cls, cfg: LogConfig) -> "EventLogger":
        return cls(
            log_path=Path(cfg.log_dir) / cfg.log_filename,
            max_age_days=getattr(cfg, "max_log_age_days", 30),
            max_size_bytes=getattr(cfg, "max_log_size_mb", 10) * 1024 * 1024,
        )

    # ------------------------------------------------------------------
    def log_event(self, confidence: float) -> int:
        """Record a confirmed snort event.  Returns the new event ID."""
        self._event_count += 1
        event_id = self._event_count

        line = (
            f"{_utc_now()}, event_id={event_id}, "
            f"confidence={confidence:.4f}, "
            f"model={self.model_version}, config={self.config_version}\n"
        )

        self._ensure_dir()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        self._maybe_rotate()
        return event_id

    # ------------------------------------------------------------------
    def read_recent(self, n: int = 20) -> list[str]:
        """Return the last *n* log lines."""
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[-n:]

    @property
    def count(self) -> int:
        return self._event_count

    def reset(self) -> None:
        """Clear the log file and reset the event counter."""
        self._event_count = 0
        if self.log_path.exists():
            self.log_path.unlink()

    # ------------------------------------------------------------------
    def _ensure_dir(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _maybe_rotate(self) -> None:
        """Rotate the log if it exceeds size or age limits."""
        if not self.log_path.exists():
            return

        # Size-based rotation
        if self.log_path.stat().st_size > self.max_size_bytes:
            self._rotate()

        # Age-based rotation
        mtime = self.log_path.stat().st_mtime
        age_days = (time.time() - mtime) / 86400.0
        if age_days > self.max_age_days:
            self._rotate()

    def _rotate(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        rotated = self.log_path.with_suffix(f".{ts}.log")
        try:
            self.log_path.rename(rotated)
        except OSError:
            pass  # rotation is best-effort

    def _count_lines(self) -> int:
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
