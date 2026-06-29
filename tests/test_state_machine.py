"""
TDD tests for the event confirmation state machine.

The state machine is the correctness layer — it decides when a
real snort event has happened and emits exactly one count per event.

States: idle → candidate → confirming → cooldown
"""

from unittest.mock import patch

import pytest

from runtime.state_machine import (
    SnortState,
    StateMachine,
)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_in_idle(self):
        sm = StateMachine()
        assert sm.state == SnortState.IDLE
        assert sm.event_count == 0

    def test_idle_rejects_low_probability(self):
        sm = StateMachine()
        counted = sm.update(probability=0.2, gate_passed=False)
        assert not counted
        assert sm.state == SnortState.IDLE
        assert sm.event_count == 0


# ---------------------------------------------------------------------------
# idle → candidate transition
# ---------------------------------------------------------------------------

class TestIdleToCandidate:
    def test_high_probability_and_gate_pass_triggers_candidate(self):
        sm = StateMachine()
        counted = sm.update(probability=0.85, gate_passed=True)
        assert not counted  # not confirmed yet
        assert sm.state == SnortState.CANDIDATE
        assert sm.event_count == 0

    def test_high_probability_but_gate_fail_enters_candidate(self):
        """Gate disagreement does not block candidate entry — classifier is primary."""
        sm = StateMachine()
        counted = sm.update(probability=0.90, gate_passed=False)
        assert not counted
        # Enters candidate because classifier says positive; gate_agreement_count = 0
        assert sm.state == SnortState.CANDIDATE

    def test_low_probability_even_with_gate_pass_stays_idle(self):
        sm = StateMachine()
        counted = sm.update(probability=0.30, gate_passed=True)
        assert not counted
        assert sm.state == SnortState.IDLE


# ---------------------------------------------------------------------------
# candidate → confirming transition
# ---------------------------------------------------------------------------

class TestCandidateToConfirming:
    def test_consecutive_positives_confirm_and_count(self):
        """N consecutive (high prob + gate passed) windows confirm a snort."""
        sm = StateMachine(confirmation_windows=3, cooldown_seconds=0.0)

        # Window 1: idle → candidate
        assert not sm.update(0.80, True)
        assert sm.state == SnortState.CANDIDATE

        # Window 2: candidate → confirming
        assert not sm.update(0.82, True)
        assert sm.state == SnortState.CONFIRMING

        # Window 3: confirming → counted
        counted = sm.update(0.79, True)
        assert counted
        assert sm.event_count == 1

    def test_single_negative_resets_confirmation(self):
        """A single failed window resets the confirmation chain."""
        sm = StateMachine(confirmation_windows=3, cooldown_seconds=0.0)

        sm.update(0.85, True)   # candidate
        sm.update(0.83, True)   # confirming
        sm.update(0.40, True)   # low prob → resets to idle
        assert sm.state == SnortState.IDLE

        # Start over — fresh chain, 3 positives → count
        sm.update(0.85, True)   # candidate
        sm.update(0.82, True)   # confirming
        sm.update(0.81, True)   # 3rd positive → counts (cooldown=0 so immediate idle)
        assert sm.event_count == 1

    def test_gate_disagreement_resets_confirmation(self):
        """If gate disagrees too often during confirmation, reset."""
        sm = StateMachine(confirmation_windows=3, min_gate_agreement=3)

        sm.update(0.85, True)   # candidate, gate agrees
        sm.update(0.83, True)   # confirming, gate agrees
        sm.update(0.81, False)  # gate disagrees → only 2 agreements < 3 → reset
        assert sm.state == SnortState.IDLE


# ---------------------------------------------------------------------------
# cooldown behavior
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_enters_cooldown_after_count(self):
        sm = StateMachine(confirmation_windows=1, cooldown_seconds=10.0, min_gate_agreement=1)

        counted = sm.update(0.85, True)
        assert counted
        assert sm.state == SnortState.COOLDOWN

    def test_ignores_events_during_cooldown(self):
        sm = StateMachine(confirmation_windows=1, cooldown_seconds=10.0, min_gate_agreement=1)

        sm.update(0.85, True)  # counted, enters cooldown
        assert sm.event_count == 1

        # During cooldown, high-confidence events are ignored
        counted = sm.update(0.95, True)
        assert not counted
        assert sm.event_count == 1
        assert sm.state == SnortState.COOLDOWN

    def test_returns_to_idle_after_cooldown(self):
        sm = StateMachine(confirmation_windows=1, cooldown_seconds=0.05, min_gate_agreement=1)

        sm.update(0.85, True)  # count 1, cooldown 0.05s
        # Wait out the cooldown
        import time
        time.sleep(0.06)

        # Now a new event can be counted
        counted = sm.update(0.85, True)
        assert counted
        assert sm.event_count == 2


# ---------------------------------------------------------------------------
# candidate timeout
# ---------------------------------------------------------------------------

class TestCandidateTimeout:
    def test_candidate_times_out_and_resets_to_idle(self):
        sm = StateMachine(candidate_timeout_seconds=0.05)

        sm.update(0.85, True)  # enters candidate
        assert sm.state == SnortState.CANDIDATE

        # Wait past timeout, then send another positive
        import time
        time.sleep(0.06)

        # Should have reset — this starts a new candidate chain
        counted = sm.update(0.85, True)
        assert sm.state == SnortState.CANDIDATE  # fresh candidate
        assert not counted


# ---------------------------------------------------------------------------
# gate agreement requirement
# ---------------------------------------------------------------------------

class TestGateAgreement:
    def test_min_gate_agreement_enforced(self):
        """Gate must agree on at least min_gate_agreement windows during confirmation."""
        sm = StateMachine(confirmation_windows=4, min_gate_agreement=3)

        # 4 consecutive positive probabilities, but gate disagrees on 2 of them
        sm.update(0.85, True)   # candidate, gate agrees
        sm.update(0.82, False)  # confirming, gate disagrees
        sm.update(0.83, True)   # still confirming, gate agrees
        sm.update(0.81, False)  # gate disagrees again — only 2 gate agreements < 3

        # Should NOT have counted
        assert sm.state == SnortState.IDLE
        assert sm.event_count == 0

    def test_gate_agreement_met_counts(self):
        sm = StateMachine(confirmation_windows=3, min_gate_agreement=2)

        sm.update(0.85, True)   # candidate
        sm.update(0.82, True)   # confirming
        counted = sm.update(0.81, False)  # confirming, gate disagrees but 2 agreements met
        assert counted
        assert sm.event_count == 1


# ---------------------------------------------------------------------------
# event counting
# ---------------------------------------------------------------------------

class TestEventCounting:
    def test_exactly_one_count_per_event(self):
        sm = StateMachine(confirmation_windows=2, cooldown_seconds=0.0, min_gate_agreement=1)

        sm.update(0.85, True)
        counted = sm.update(0.83, True)
        assert counted

        # The very next window after counting should NOT count again
        # (cooldown prevents this even at 0.0 — it's immediate-idle in test)
        sm.update(0.90, True)  # fresh candidate
        assert sm.event_count == 1  # hasn't confirmed 2nd event yet

    def test_multiple_events_over_time(self):
        sm = StateMachine(confirmation_windows=1, cooldown_seconds=0.01, min_gate_agreement=1)

        for i in range(5):
            counted = sm.update(0.85, True)
            assert counted
            assert sm.event_count == i + 1
            if sm.cooldown_seconds > 0:
                import time
                time.sleep(0.02)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_probability_exactly_at_threshold(self):
        sm = StateMachine(confirmation_windows=1, min_gate_agreement=1)
        # Default threshold is 0.70 from config
        counted = sm.update(probability=0.70, gate_passed=True)
        assert counted  # >= threshold

    def test_probability_just_below_threshold(self):
        sm = StateMachine(confirmation_windows=2)
        sm.update(probability=0.85, gate_passed=True)  # candidate
        # Next window: prob just below threshold with gate true
        sm.update(probability=0.69, gate_passed=True)  # should reset
        assert sm.state == SnortState.IDLE

    def test_reset_clears_count(self):
        sm = StateMachine(confirmation_windows=1, cooldown_seconds=0.0, min_gate_agreement=1)
        sm.update(0.85, True)
        assert sm.event_count == 1
        sm.reset()
        assert sm.event_count == 0
        assert sm.state == SnortState.IDLE

    def test_long_silence_stays_idle(self):
        sm = StateMachine()
        for _ in range(1000):
            counted = sm.update(0.0, False)
            assert not counted
        assert sm.state == SnortState.IDLE
        assert sm.event_count == 0

    def test_rapid_alternating_noise(self):
        """Rapidly alternating pass/fail should not produce false counts."""
        sm = StateMachine(confirmation_windows=5)
        for i in range(100):
            alt_pass = (i % 2 == 0)
            counted = sm.update(0.80 if alt_pass else 0.10, alt_pass)
            assert not counted  # never sustains 5 consecutive
        assert sm.event_count == 0
