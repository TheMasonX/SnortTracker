"""
Event confirmation state machine for the SnortTracker runtime.

This is the *correctness layer*.  The model supplies evidence; the
state machine decides whether a real snort event has occurred and
emits exactly one count per event.

States
------
idle        — waiting for a candidate
candidate   — first positive window seen, watching for more
confirming  — accumulating evidence toward a count
cooldown    — refractory period after a count (prevents double-counting)

Within ``cooldown``, all incoming evidence is ignored.  After the
cooldown timer expires the state returns to ``idle`` automatically
on the next ``update()`` call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from runtime.config import StateMachineConfig


class SnortState(Enum):
    IDLE = auto()
    CANDIDATE = auto()
    CONFIRMING = auto()
    COOLDOWN = auto()


@dataclass
class StateMachine:
    """Deterministic confirmation of snort events.

    Parameters
    ----------
    confirmation_windows : int
        Number of consecutive positive windows required to confirm a snort.
    cooldown_seconds : float
        Refractory period after a count during which all evidence is ignored.
    candidate_timeout_seconds : float
        Max time spent in CANDIDATE before resetting to IDLE.
    min_gate_agreement : int
        Minimum number of windows during confirmation where the gate must
        agree with the classifier.  Prevents gate failures from being ignored.
    probability_threshold : float
        Minimum classifier confidence [0, 1] for a window to be considered positive.
    """

    # Tuning parameters
    confirmation_windows: int = 3
    cooldown_seconds: float = 1.5
    candidate_timeout_seconds: float = 0.8
    min_gate_agreement: int = 2
    probability_threshold: float = 0.70

    # Internal state
    state: SnortState = SnortState.IDLE
    event_count: int = 0
    _confirmation_count: int = 0
    _gate_agreement_count: int = 0
    _state_entered_at: float = 0.0
    _cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        confirmation_windows: int = 3,
        cooldown_seconds: float = 1.5,
        candidate_timeout_seconds: float = 0.8,
        min_gate_agreement: int = 2,
        probability_threshold: float = 0.70,
    ) -> None:
        self.confirmation_windows = confirmation_windows
        self.cooldown_seconds = cooldown_seconds
        self.candidate_timeout_seconds = candidate_timeout_seconds
        self.min_gate_agreement = min_gate_agreement
        self.probability_threshold = probability_threshold

        self.state = SnortState.IDLE
        self.event_count = 0
        self._confirmation_count = 0
        self._gate_agreement_count = 0
        self._state_entered_at = time.monotonic()
        self._cooldown_until = 0.0

    @classmethod
    def from_config(cls, cfg: StateMachineConfig) -> "StateMachine":
        return cls(
            confirmation_windows=cfg.confirmation_windows,
            cooldown_seconds=cfg.cooldown_seconds,
            candidate_timeout_seconds=cfg.candidate_timeout_seconds,
            min_gate_agreement=cfg.min_gate_agreement,
        )

    # ------------------------------------------------------------------
    def update(self, probability: float, gate_passed: bool) -> bool:
        """Process one inference window and return True if a snort is counted."""
        now = time.monotonic()

        # --- cooldown: ignore everything ---
        if self.state == SnortState.COOLDOWN:
            if now >= self._cooldown_until:
                self._transition_to(SnortState.IDLE, now)
            else:
                return False

        # --- evaluate this window ---
        # Classifier decides "positive"; gate is an independent agreement signal
        is_positive = probability >= self.probability_threshold

        if self.state == SnortState.IDLE:
            if is_positive:
                self._confirmation_count = 1
                self._gate_agreement_count = 1 if gate_passed else 0

                # Check immediate-confirm case (confirmation_windows == 1)
                if self._confirmation_count >= self.confirmation_windows:
                    if self._gate_agreement_count >= self.min_gate_agreement:
                        return self._emit_count(now)
                    self._confirmation_count = 0
                    self._gate_agreement_count = 0
                    return False

                self._transition_to(SnortState.CANDIDATE, now)
            return False

        if self.state == SnortState.CANDIDATE:
            # Timeout check
            if now - self._state_entered_at > self.candidate_timeout_seconds:
                self._transition_to(SnortState.IDLE, now)
                self._confirmation_count = 0
                self._gate_agreement_count = 0
                return self.update(probability, gate_passed)

            if is_positive:
                self._confirmation_count += 1
                if gate_passed:
                    self._gate_agreement_count += 1

                if self._confirmation_count >= self.confirmation_windows:
                    if self._gate_agreement_count >= self.min_gate_agreement:
                        return self._emit_count(now)
                    self._transition_to(SnortState.IDLE, now)
                    self._confirmation_count = 0
                    self._gate_agreement_count = 0
                    return False

                self._transition_to(SnortState.CONFIRMING, now)
                return False
            else:
                self._transition_to(SnortState.IDLE, now)
                self._confirmation_count = 0
                self._gate_agreement_count = 0
                return False

        if self.state == SnortState.CONFIRMING:
            if is_positive:
                self._confirmation_count += 1
                if gate_passed:
                    self._gate_agreement_count += 1

                if self._confirmation_count >= self.confirmation_windows:
                    if self._gate_agreement_count >= self.min_gate_agreement:
                        return self._emit_count(now)
                    else:
                        self._transition_to(SnortState.IDLE, now)
                        self._confirmation_count = 0
                        self._gate_agreement_count = 0
                        return False

                # Stay in confirming
                return False
            else:
                # Negative — reset
                self._transition_to(SnortState.IDLE, now)
                self._confirmation_count = 0
                self._gate_agreement_count = 0
                return False

        return False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset event count and state to initial."""
        self.state = SnortState.IDLE
        self.event_count = 0
        self._confirmation_count = 0
        self._gate_agreement_count = 0
        self._state_entered_at = time.monotonic()
        self._cooldown_until = 0.0

    # ------------------------------------------------------------------
    def _emit_count(self, now: float) -> bool:
        self.event_count += 1
        self._transition_to(SnortState.COOLDOWN, now)
        self._cooldown_until = now + self.cooldown_seconds
        self._confirmation_count = 0
        self._gate_agreement_count = 0
        return True

    def _transition_to(self, new_state: SnortState, now: float) -> None:
        self.state = new_state
        self._state_entered_at = now
