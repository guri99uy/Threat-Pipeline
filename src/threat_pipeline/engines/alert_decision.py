"""Engine 6 — Alert fusion and incident/cooldown logic.

C#2/C#6 upgrade: Three-input fusion with continuous signals, UNCERTAIN state,
configurable thresholds, IncidentState escalation boost.

Combines:
  - LLM threat score × ASR confidence (C#3)
  - Continuous audio features (C#5)
  - DistilRoBERTa toxicity (C#7, parallel branch)
  - YAMNet sound events (C#9, non-decisive)
  - IncidentState escalation (C#1)

Production note: weights and thresholds would be tuned on labelled data.
"""

from __future__ import annotations

import time
import uuid

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import (
    AlertAction,
    AlertDecision,
    AudioFeatures,
    IncidentSnapshot,
    SoundEventResult,
    TextClassification,
    ThreatAssessment,
    Wav2Vec2Result,
)


class AlertDecisionInput:
    def __init__(
        self,
        threat: ThreatAssessment,
        features: AudioFeatures,
        source_id: str = "",
        text_classification: TextClassification | None = None,
        wav2vec2_result: Wav2Vec2Result | None = None,
        sound_events: SoundEventResult | None = None,
        asr_confidence: float = 1.0,
        incident_snapshot: IncidentSnapshot | None = None,
    ):
        self.threat = threat
        self.features = features
        self.source_id = source_id
        self.text_classification = text_classification
        self.wav2vec2_result = wav2vec2_result
        self.sound_events = sound_events
        self.asr_confidence = asr_confidence
        self.incident_snapshot = incident_snapshot


# Threat keywords that boost the score
_BOOST_KEYWORDS = {
    "kill", "hurt", "die", "stab", "shoot", "gun", "knife",
    "weapon", "bomb", "attack", "murder", "destroy",
}


class AlertDecisionEngine(Engine[AlertDecisionInput, AlertDecision]):
    """Weighted fusion of threat + audio + classifier signals → alert action."""

    name = "alert_decision"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        # Cooldown state: source_key → (last_alert_time, incident_id)
        self._cooldowns: dict[str, tuple[float, str]] = {}

    def process(self, input_data: AlertDecisionInput) -> AlertDecision:
        threat = input_data.threat
        features = input_data.features
        s = self.settings

        # ── C#3: Effective threat = threat_score × asr_confidence ───────
        effective_threat = threat.threat_score * input_data.asr_confidence

        # ── C#5: Continuous loudness (replaces binary) ──────────────────
        loudness_continuous = features.rms_normalized

        # Keyword boost (continuous: 1.0 if match, 0.0 otherwise)
        detected_lower = [kw.lower() for kw in threat.keywords_detected]
        has_keyword = any(
            boost in phrase
            for phrase in detected_lower
            for boost in _BOOST_KEYWORDS
        )
        keyword_boost = 1.0 if has_keyword else 0.0

        # ── C#4: Directed boost weighted by confidence_in_direction ─────
        directed_boost = s.directed_boost * threat.confidence_in_direction if threat.is_directed else 0.0

        # ── Base fusion ─────────────────────────────────────────────────
        base_score = (
            s.weight_threat * effective_threat
            + s.weight_loudness * loudness_continuous
            + s.weight_keyword * keyword_boost
            + directed_boost
        )

        # ── C#7: DistilRoBERTa parallel branch ─────────────────────────
        if input_data.text_classification and s.weight_text_classifier > 0:
            base_score += s.weight_text_classifier * input_data.text_classification.toxicity_score

        # ── C#9: YAMNet non-decisive signal ─────────────────────────────
        if input_data.sound_events and input_data.sound_events.has_threat_sound:
            base_score += s.yamnet_weight_in_fusion * input_data.sound_events.max_threat_event_confidence

        final_score = max(0.0, min(1.0, base_score))

        # ── Action mapping (thresholds from config) ─────────────────────
        if final_score >= s.threshold_escalate:
            action = AlertAction.ESCALATE
        elif final_score >= s.threshold_alert:
            action = AlertAction.ALERT
        elif s.uncertain_score_low <= final_score <= s.uncertain_score_high:
            action = AlertAction.UNCERTAIN
        elif final_score >= s.threshold_log:
            action = AlertAction.LOG
        else:
            action = AlertAction.NO_ACTION

        # ── C#1: Escalation boost from IncidentState ───────────────────
        if input_data.incident_snapshot and input_data.incident_snapshot.is_escalating:
            action = self._promote_action(action)

        # ── Cooldown / incident grouping ────────────────────────────────
        now = time.time()
        source_key = input_data.source_id if input_data.source_id else f"turn_{threat.turn_id}"
        suppressed = False
        incident_id = uuid.uuid4().hex[:12]

        if source_key in self._cooldowns:
            last_time, existing_id = self._cooldowns[source_key]
            if now - last_time < s.cooldown_seconds:
                suppressed = True
                incident_id = existing_id

        if action in (AlertAction.ALERT, AlertAction.ESCALATE) and not suppressed:
            self._cooldowns[source_key] = (now, incident_id)

        return AlertDecision(
            turn_id=threat.turn_id,
            action=action,
            final_score=final_score,
            suppressed=suppressed,
            incident_id=incident_id,
        )

    @staticmethod
    def _promote_action(action: AlertAction) -> AlertAction:
        """Promote an action by one tier due to escalation."""
        promotion = {
            AlertAction.NO_ACTION: AlertAction.LOG,
            AlertAction.LOG: AlertAction.UNCERTAIN,
            AlertAction.UNCERTAIN: AlertAction.ALERT,
            AlertAction.ALERT: AlertAction.ESCALATE,
            AlertAction.ESCALATE: AlertAction.ESCALATE,
        }
        return promotion[action]
