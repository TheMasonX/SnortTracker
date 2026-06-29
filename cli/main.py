"""
CLI entry point for SnortTracker.

Commands: start, view, reset, tail, status, purge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from runtime.capture import WavFileCapture, create_capture_source
from runtime.classifier import Classifier, create_classifier
from runtime.config import config
from runtime.gating import Gate
from runtime.logging import EventLogger
from runtime.state_machine import StateMachine


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _get_logger() -> EventLogger:
    return EventLogger.from_config(config.log)


def _build_pipeline(wav_path: Path | None = None):
    """Construct capture → gate → classifier → state machine → logger."""
    cap = create_capture_source(wav_path=wav_path)
    gate = Gate(config.gate)
    clf = create_classifier(config.inference)
    sm = StateMachine.from_config(config.state_machine)
    logger = _get_logger()
    return cap, gate, clf, sm, logger


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> int:
    """Begin the continuous listening loop."""
    wav_path = Path(args.input) if args.input else None
    cap, gate, clf, sm, logger = _build_pipeline(wav_path)

    print(f"SnortTracker listening... (Ctrl+C to stop)")
    print(f"  Model: {clf.model_name} v{clf.model_version}")
    if wav_path:
        print(f"  Dry-run mode: {wav_path}")

    window_count = 0
    gate_passes = 0

    try:
        while True:
            window = cap.get_window()
            if window is None:
                break  # end of WAV file

            window_count += 1
            result = gate.evaluate(window.audio)
            if result.passed:
                gate_passes += 1

            probability = clf.predict(window.audio, result)
            counted = sm.update(
                probability=probability,
                gate_passed=result.passed,
            )

            if counted:
                event_id = logger.log_event(
                    confidence=probability,
                )
                ts = window.timestamp_utc.isoformat(timespec="seconds")
                print(f"  🐽 Snort #{event_id} detected at {ts} "
                      f"(confidence={probability:.3f})")

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        if hasattr(cap, "close"):
            cap.close()
        if hasattr(cap, "stop"):
            cap.stop()

    print(f"Windows processed: {window_count}")
    print(f"Gate passes: {gate_passes}")
    print(f"Snorts counted: {sm.event_count}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    """Show the current snort count."""
    logger = _get_logger()
    print(f"Snort count: {logger.count}" if logger.count > 0 else "Snort count: 0")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Clear the count and log file."""
    logger = _get_logger()
    logger.reset()
    print("Count and logs cleared.")
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    """Show recent events."""
    n = getattr(args, "lines", 20)
    logger = _get_logger()
    lines = logger.read_recent(n)
    if not lines:
        print("No events logged yet.")
    else:
        sys.stdout.writelines(lines)
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    """Wipe all logs and rotated archives with confirmation."""
    logger = _get_logger()

    if not args.yes:
        confirm = input(
            "This will delete ALL snort logs and rotated archives. "
            "Continue? [y/N] "
        )
        if confirm.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    removed = _purge_logs(logger)
    if removed:
        print(f"Purged {len(removed)} log file(s):")
        for path in removed:
            print(f"  {path}")
    else:
        print("No log files to purge.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show runtime health."""
    logger = _get_logger()
    print(f"Log file: {logger.log_path}")
    print(f"Log exists: {logger.log_path.exists()}")
    if logger.log_path.exists():
        size_kb = logger.log_path.stat().st_size / 1024
        print(f"Log size: {size_kb:.1f} KB")
    print(f"Recorded events: {logger.count}")
    print(f"Model: {logger.model_version}")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _purge_logs(logger: EventLogger) -> list[Path]:
    """Delete the main log, all rotated archives, and the count file.

    Returns the list of removed files.
    """
    removed: list[Path] = []

    # Main log
    if logger.log_path.exists():
        logger.log_path.unlink()
        removed.append(logger.log_path)

    # Rotated archives (*.YYYYMMDDTHHMMSS.log)
    log_dir = logger.log_path.parent
    if log_dir.exists():
        pattern = logger.log_path.stem + ".*.log"
        for archive in sorted(log_dir.glob(pattern)):
            archive.unlink()
            removed.append(archive)

    # Reset in-memory count
    logger.reset()
    return removed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="snort-counter",
        description="SnortTracker — a playful snort counter",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = sub.add_parser("start", help="Begin listening for snorts")
    p_start.add_argument(
        "--input", type=str, default=None,
        help="WAV file for dry-run mode (omit for live microphone)",
    )
    p_start.set_defaults(func=cmd_start)

    # view
    p_view = sub.add_parser("view", help="Show current snort count")
    p_view.set_defaults(func=cmd_view)

    # reset
    p_reset = sub.add_parser("reset", help="Clear count and logs")
    p_reset.set_defaults(func=cmd_reset)

    # tail
    p_tail = sub.add_parser("tail", help="Show recent events")
    p_tail.add_argument("-n", "--lines", type=int, default=20)
    p_tail.set_defaults(func=cmd_tail)

    # status
    p_status = sub.add_parser("status", help="Show runtime health")
    p_status.set_defaults(func=cmd_status)

    # purge
    p_purge = sub.add_parser("purge", help="Wipe all logs and rotated archives")
    p_purge.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    p_purge.set_defaults(func=cmd_purge)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
