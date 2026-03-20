"""Tests for the AlertPublisher engine and EventBus."""

import pytest

from threat_pipeline.engines.alert_publisher import AlertPublisherEngine, AlertPublisherInput
from threat_pipeline.event_bus import EventBus
from threat_pipeline.models import (
    AlertAction,
    AlertDecision,
    AlertEvent,
    AudioFeatures,
    ThreatAssessment,
    ThreatLevel,
    TranscriptionResult,
)


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        bus.publish("test", {"msg": "hello"})

        assert len(received) == 1
        assert received[0]["msg"] == "hello"

    def test_history(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert len(bus.history) == 2

    def test_clear(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.clear()
        assert len(bus.history) == 0


class TestAlertPublisher:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def engine(self, bus):
        return AlertPublisherEngine(bus)

    def _make_input(self, action: AlertAction, suppressed: bool = False) -> AlertPublisherInput:
        return AlertPublisherInput(
            decision=AlertDecision(
                turn_id=0, action=action, final_score=0.85, suppressed=suppressed
            ),
            transcription=TranscriptionResult(
                turn_id=0, raw_text="test", cleaned_text="test",
                asr_confidence=0.9, no_speech_prob=0.01,
            ),
            threat=ThreatAssessment(
                turn_id=0,
                threat_level=ThreatLevel.HIGH,
                threat_score=0.85,
                is_directed=True,
                reasoning="Direct threat",
                keywords_detected=["hurt"],
                confidence_in_direction=0.9,
            ),
            features=AudioFeatures(
                turn_id=0,
                rms_db=-15.0,
                spectral_centroid_mean=3500.0,
                zero_crossing_rate=0.05,
                is_loud=True,
                is_sharp=True,
                rms_normalized=0.85,
                centroid_normalized=0.73,
            ),
            latency_breakdown={"transcription": 0.5},
        )

    def test_publishes_alert(self, engine, bus):
        inp = self._make_input(AlertAction.ALERT)
        event = engine.run(inp)

        assert event is not None
        assert isinstance(event, AlertEvent)
        assert len(bus.history) == 1

    def test_publishes_escalate(self, engine, bus):
        inp = self._make_input(AlertAction.ESCALATE)
        event = engine.run(inp)
        assert event is not None

    def test_no_publish_for_log(self, engine, bus):
        inp = self._make_input(AlertAction.LOG)
        event = engine.run(inp)
        assert event is None
        assert len(bus.history) == 0

    def test_no_publish_for_no_action(self, engine, bus):
        inp = self._make_input(AlertAction.NO_ACTION)
        event = engine.run(inp)
        assert event is None

    def test_uncertain_publishes_to_review_topic(self, engine, bus):
        """UNCERTAIN should publish to 'review' topic, not 'alerts'."""
        inp = self._make_input(AlertAction.UNCERTAIN)
        event = engine.run(inp)

        assert event is None  # No AlertEvent returned
        # But review topic should have been published to
        review_events = [e for topic, e in bus.history if topic == "review"]
        assert len(review_events) == 1
        assert review_events[0]["action"] == "uncertain"

    def test_suppressed_not_published(self, engine, bus):
        inp = self._make_input(AlertAction.ALERT, suppressed=True)
        event = engine.run(inp)
        assert event is None
        assert len(bus.history) == 0

    def test_extended_indicators(self, engine, bus):
        """Published events should include extended indicators."""
        inp = self._make_input(AlertAction.ALERT)
        event = engine.run(inp)

        assert event is not None
        indicators = event.audio_indicators
        assert "rms_normalized" in indicators
        assert "centroid_normalized" in indicators
        assert "asr_confidence" in indicators
        assert "confidence_in_direction" in indicators
