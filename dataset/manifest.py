"""
Dataset manifest for SnortTracker.

Tracks every audio recording, its session, label status, and assigned
data split (train/val/test).  The manifest is the single source of
truth for what data exists and how it should be used.

Format (CSV)
------------
session_id, path, duration_seconds, label_status, split, notes

- ``label_status``: ``unlabeled`` | ``positive`` | ``negative`` | ``ignore``
- ``split``: ``train`` | ``val`` | ``test`` | ``unassigned``
- ``notes``: free-text (e.g., "confirmed snort at 1.2s, 3.4s")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional


class LabelStatus(str, Enum):
    UNLABELED = "unlabeled"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    IGNORE = "ignore"


class DataSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNASSIGNED = "unassigned"


@dataclass
class ManifestEntry:
    """One row in the dataset manifest."""

    session_id: str
    path: Path
    duration_seconds: float
    label_status: LabelStatus = LabelStatus.UNLABELED
    split: DataSplit = DataSplit.UNASSIGNED
    notes: str = ""


@dataclass
class Manifest:
    """Read/write interface for the dataset manifest CSV.

    Usage::

        m = Manifest("manifest.csv")
        m.add(session_id="S001", path=Path("recordings/s001.wav"), duration=30.0)
        m.label("S001", LabelStatus.POSITIVE, notes="confirmed snort")
        m.assign_split("S001", DataSplit.TRAIN)
        m.save()
    """

    path: Path
    _entries: dict[str, ManifestEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        session_id: str,
        path: Path,
        duration_seconds: float,
        label_status: LabelStatus = LabelStatus.UNLABELED,
        split: DataSplit = DataSplit.UNASSIGNED,
        notes: str = "",
    ) -> ManifestEntry:
        """Add or overwrite a manifest entry."""
        entry = ManifestEntry(
            session_id=session_id,
            path=Path(path),
            duration_seconds=duration_seconds,
            label_status=label_status,
            split=split,
            notes=notes,
        )
        self._entries[session_id] = entry
        return entry

    def get(self, session_id: str) -> Optional[ManifestEntry]:
        return self._entries.get(session_id)

    def remove(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def label(
        self,
        session_id: str,
        status: LabelStatus,
        notes: str = "",
    ) -> None:
        """Assign a label to a session."""
        entry = self._entries.get(session_id)
        if entry is None:
            raise KeyError(f"Session not found: {session_id}")
        entry.label_status = status
        if notes:
            entry.notes = notes

    def assign_split(self, session_id: str, split: DataSplit) -> None:
        """Assign a data split to a session."""
        entry = self._entries.get(session_id)
        if entry is None:
            raise KeyError(f"Session not found: {session_id}")
        entry.split = split

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[ManifestEntry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[ManifestEntry]:
        return list(self._entries.values())

    def by_label(self, status: LabelStatus) -> list[ManifestEntry]:
        return [e for e in self._entries.values() if e.label_status == status]

    def by_split(self, split: DataSplit) -> list[ManifestEntry]:
        return [e for e in self._entries.values() if e.split == split]

    def labeled(self) -> list[ManifestEntry]:
        """Entries with a definitive label (not unlabeled or ignore)."""
        return [
            e for e in self._entries.values()
            if e.label_status in (LabelStatus.POSITIVE, LabelStatus.NEGATIVE)
        ]

    @property
    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._entries.values():
            key = e.label_status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    COLUMNS = [
        "session_id", "path", "duration_seconds",
        "label_status", "split", "notes",
    ]

    def save(self) -> None:
        """Write the manifest to CSV."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for entry in sorted(
                self._entries.values(),
                key=lambda e: e.session_id,
            ):
                writer.writerow([
                    entry.session_id,
                    str(entry.path),
                    f"{entry.duration_seconds:.3f}",
                    entry.label_status.value,
                    entry.split.value,
                    entry.notes,
                ])

    def _load(self) -> None:
        """Read the manifest from CSV."""
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["session_id"]
                self._entries[sid] = ManifestEntry(
                    session_id=sid,
                    path=Path(row["path"]),
                    duration_seconds=float(row["duration_seconds"]),
                    label_status=LabelStatus(row["label_status"]),
                    split=DataSplit(row["split"]),
                    notes=row.get("notes", ""),
                )
