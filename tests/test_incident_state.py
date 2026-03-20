"""Tests for the IncidentState tracker."""

import pytest

from threat_pipeline.config import Settings
from threat_pipeline.incident_state import IncidentState


class TestIncidentState:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test",
            incident_ema_alpha=0.4,
            escalation_consecutive_threshold=3,
        )

    @pytest.fixture
    def state(self, settings):
        return IncidentState(settings)

    def test_first_observation_seeds_ema(self, state):
        snap = state.update("src1", 0.8)
        assert snap.accumulated_score == pytest.approx(0.8)
        assert snap.turn_count == 1
        assert snap.consecutive_rising == 0

    def test_ema_computation(self, state):
        """EMA = α × current + (1-α) × prior."""
        state.update("src1", 0.5)  # seeds at 0.5
        snap = state.update("src1", 0.9)
        # 0.4 * 0.9 + 0.6 * 0.5 = 0.36 + 0.30 = 0.66
        assert snap.accumulated_score == pytest.approx(0.66, abs=0.01)

    def test_consecutive_rising_detection(self, state):
        state.update("src1", 0.2)
        state.update("src1", 0.3)  # rising → 1
        state.update("src1", 0.4)  # rising → 2
        snap = state.update("src1", 0.5)  # rising → 3 → escalating!
        assert snap.consecutive_rising == 3
        assert snap.is_escalating is True

    def test_consecutive_resets_on_drop(self, state):
        state.update("src1", 0.2)
        state.update("src1", 0.4)  # rising → 1
        state.update("src1", 0.3)  # drop → 0
        snap = state.update("src1", 0.5)  # rising → 1
        assert snap.consecutive_rising == 1
        assert snap.is_escalating is False

    def test_different_source_isolation(self, state):
        """Sources should be tracked independently."""
        state.update("src1", 0.9)
        snap_b = state.update("src2", 0.1)

        assert snap_b.accumulated_score == pytest.approx(0.1)
        assert snap_b.source_id == "src2"

        snap_a = state.get_state("src1")
        assert snap_a is not None
        assert snap_a.accumulated_score == pytest.approx(0.9)

    def test_get_state_returns_none_for_unknown(self, state):
        assert state.get_state("unknown") is None

    def test_not_escalating_below_threshold(self, state):
        state.update("src1", 0.1)
        snap = state.update("src1", 0.2)  # 1 consecutive
        assert snap.is_escalating is False

    def test_configurable_threshold(self):
        settings = Settings(
            openai_api_key="test",
            incident_ema_alpha=0.5,
            escalation_consecutive_threshold=2,
        )
        state = IncidentState(settings)
        state.update("s", 0.1)
        state.update("s", 0.2)
        snap = state.update("s", 0.3)
        assert snap.consecutive_rising == 2
        assert snap.is_escalating is True
