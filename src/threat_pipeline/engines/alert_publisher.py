"""Engine 7 — Publish structured alert events.

Converts an AlertDecision (plus context) into a full AlertEvent and
publishes it to the EventBus.  Only ALERT and ESCALATE actions are
published; UNCERTAIN is logged for human review but not published
to the "alerts" topic.  LOG and NO_ACTION are silently recorded.

Production note: the EventBus would be replaced by Redis Pub/Sub or
a message queue delivering to the ARC operator console.
"""

from __future__ import annotations

from typing import Optional

from threat_pipeline.engine_base import Engine
from threat_pipeline.event_bus import EventBus
from threat_pipeline.models import (
    AlertAction,
    AlertDecision,
    AlertEvent,
    AudioFeatures,
    IncidentSnapshot,
    SoundEventResult,
    TextClassification,
    ThreatAssessment,
    TranscriptionResult,
    Wav2Vec2Result,
)


class AlertPublisherInput:
    def __init__(
        self,
        decision: AlertDecision,
        transcription: TranscriptionResult,
        threat: ThreatAssessment,
        features: AudioFeatures,
        latency_breakdown: dict[str, float],
        text_classification: Optional[TextClassification] = None,
        wav2vec2_result: Optional[Wav2Vec2Result] = None,
        sound_events: Optional[SoundEventResult] = None,
        incident_snapshot: Optional[IncidentSnapshot] = None,
    ):
        self.decision = decision
        self.transcription = transcription
        self.threat = threat
        self.features = features
        self.latency_breakdown = latency_breakdown
        self.text_classification = text_classification
        self.wav2vec2_result = wav2vec2_result
        self.sound_events = sound_events
        self.incident_snapshot = incident_snapshot


class AlertPublisherEngine(Engine[AlertPublisherInput, AlertEvent | None]):
    """Build and publish AlertEvents for actionable decisions."""

    name = "alert_publisher"

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__()
        self.event_bus = event_bus

    def process(self, input_data: AlertPublisherInput) -> AlertEvent | None:
        d = input_data.decision

        # Only publish for actionable alerts
        if d.action not in (AlertAction.ALERT, AlertAction.ESCALATE):
            # Log UNCERTAIN to review topic (but don't return an AlertEvent)
            if d.action == AlertAction.UNCERTAIN:
                self._publish_uncertain_review(input_data)
            return None

        if d.suppressed:
            return None

        indicators = self._build_indicators(input_data)

        event = AlertEvent(
            incident_id=d.incident_id,
            turn_id=d.turn_id,
            action=d.action,
            final_score=d.final_score,
            transcript=input_data.transcription.cleaned_text,
            reasoning=input_data.threat.reasoning,
            keywords=input_data.threat.keywords_detected,
            audio_indicators=indicators,
            latency_breakdown=input_data.latency_breakdown,
        )

        self.event_bus.publish("alerts", event)
        return event

    def _build_indicators(self, input_data: AlertPublisherInput) -> dict:
        """Build extended audio_indicators dict with all available signals."""
        indicators: dict = {
            "rms_db": input_data.features.rms_db,
            "is_loud": input_data.features.is_loud,
            "is_sharp": input_data.features.is_sharp,
            "spectral_centroid_mean": input_data.features.spectral_centroid_mean,
            "rms_normalized": input_data.features.rms_normalized,
            "centroid_normalized": input_data.features.centroid_normalized,
            "asr_confidence": input_data.transcription.asr_confidence,
            "confidence_in_direction": input_data.threat.confidence_in_direction,
        }

        if input_data.text_classification:
            indicators["text_classifier_score"] = input_data.text_classification.toxicity_score
            indicators["text_classifier_label"] = input_data.text_classification.label

        if input_data.wav2vec2_result:
            indicators["wav2vec2_transcript"] = input_data.wav2vec2_result.ctc_transcript

        if input_data.sound_events:
            indicators["yamnet_events"] = input_data.sound_events.events
            indicators["yamnet_has_threat"] = input_data.sound_events.has_threat_sound

        if input_data.incident_snapshot:
            indicators["incident_accumulated_score"] = input_data.incident_snapshot.accumulated_score
            indicators["incident_is_escalating"] = input_data.incident_snapshot.is_escalating
            indicators["incident_consecutive_rising"] = input_data.incident_snapshot.consecutive_rising

        return indicators

    def _publish_uncertain_review(self, input_data: AlertPublisherInput) -> None:
        """Publish UNCERTAIN decisions to a separate topic for human review."""
        review_data = {
            "turn_id": input_data.decision.turn_id,
            "final_score": input_data.decision.final_score,
            "transcript": input_data.transcription.cleaned_text,
            "reasoning": input_data.threat.reasoning,
            "action": "uncertain",
        }
        self.event_bus.publish("review", review_data)
