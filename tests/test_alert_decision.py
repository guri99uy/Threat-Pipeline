"""Tests for the AlertDecision engine."""

import time

import pytest

from threat_pipeline.engines.alert_decision import AlertDecisionEngine, AlertDecisionInput
from threat_pipeline.models import (
    AlertAction,
    AudioFeatures,
    IncidentSnapshot,
    SoundEventResult,
    TextClassification,
    ThreatAssessment,
    ThreatLevel,
)


class TestAlertDecision:
    @pytest.fixture
    def engine(self, settings):
        return AlertDecisionEngine(settings)

    def test_high_threat_loud_keywords_escalate(self, engine, sample_threat_high, sample_features):
        """High threat + loud + keywords → ESCALATE or ALERT."""
        inp = AlertDecisionInput(
            sample_threat_high, sample_features,
            asr_confidence=0.85,
        )
        decision = engine.run(inp)

        assert decision.action in (AlertAction.ALERT, AlertAction.ESCALATE)
        assert decision.final_score >= 0.7

    def test_no_threat_quiet_no_action(self, engine, sample_threat_none, benign_features):
        """No threat + quiet + no keywords → NO_ACTION."""
        inp = AlertDecisionInput(
            sample_threat_none, benign_features,
            asr_confidence=0.9,
        )
        decision = engine.run(inp)

        assert decision.action == AlertAction.NO_ACTION
        assert decision.final_score < 0.4

    def test_uncertain_state(self, engine):
        """Score in the uncertain range should produce UNCERTAIN."""
        threat = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.MEDIUM, threat_score=0.95,
            is_directed=False, reasoning="test", keywords_detected=[],
            confidence_in_direction=0.5,
        )
        features = AudioFeatures(
            turn_id=0, rms_db=-25.0, spectral_centroid_mean=2000.0,
            zero_crossing_rate=0.05, is_loud=False, is_sharp=False,
            rms_normalized=0.3, centroid_normalized=0.3,
        )
        # 0.6*0.95*1.0 + 0.2*0.3 + 0 + 0 = 0.57 + 0.06 = 0.63
        inp = AlertDecisionInput(threat, features, asr_confidence=1.0)
        decision = engine.run(inp)

        assert decision.action == AlertAction.UNCERTAIN

    def test_fusion_formula_with_asr_confidence(self, engine, settings):
        """Effective threat = threat_score × asr_confidence."""
        threat = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.HIGH, threat_score=0.8,
            is_directed=True, reasoning="test", keywords_detected=["hurt"],
            confidence_in_direction=0.9,
        )
        features = AudioFeatures(
            turn_id=0, rms_db=-15.0, spectral_centroid_mean=3500.0,
            zero_crossing_rate=0.05, is_loud=True, is_sharp=True,
            rms_normalized=0.85, centroid_normalized=0.73,
        )
        # asr_confidence=0.5 should attenuate threat
        inp = AlertDecisionInput(threat, features, asr_confidence=0.5)
        decision_low = engine.run(inp)

        engine2 = AlertDecisionEngine(settings)
        inp2 = AlertDecisionInput(threat, features, asr_confidence=1.0)
        decision_high = engine2.run(inp2)

        assert decision_low.final_score < decision_high.final_score

    def test_escalation_boost(self, engine, sample_features):
        """IncidentState escalation should promote action tier."""
        threat = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.MEDIUM, threat_score=0.5,
            is_directed=False, reasoning="test", keywords_detected=[],
            confidence_in_direction=0.5,
        )
        # Without escalation
        inp1 = AlertDecisionInput(threat, sample_features, asr_confidence=1.0)
        d1 = engine.run(inp1)

        # With escalation
        escalating_snap = IncidentSnapshot(
            source_id="test", accumulated_score=0.7,
            consecutive_rising=5, turn_count=6, is_escalating=True,
        )
        engine2 = AlertDecisionEngine(engine.settings)
        inp2 = AlertDecisionInput(
            threat, sample_features, asr_confidence=1.0,
            incident_snapshot=escalating_snap,
        )
        d2 = engine2.run(inp2)

        # Escalating should produce equal or higher tier
        action_order = list(AlertAction)
        assert action_order.index(d2.action) >= action_order.index(d1.action)

    def test_yamnet_boost(self, engine, sample_features):
        """YAMNet threat sound should boost the score."""
        threat = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.MEDIUM, threat_score=0.6,
            is_directed=False, reasoning="test", keywords_detected=[],
            confidence_in_direction=0.5,
        )
        sound_events = SoundEventResult(
            turn_id=0,
            events=[{"class": "Gunshot", "confidence": 0.9}],
            max_threat_event_confidence=0.9,
            has_threat_sound=True,
        )

        inp_no_yamnet = AlertDecisionInput(threat, sample_features, asr_confidence=1.0)
        d1 = engine.run(inp_no_yamnet)

        engine2 = AlertDecisionEngine(engine.settings)
        inp_yamnet = AlertDecisionInput(
            threat, sample_features, asr_confidence=1.0,
            sound_events=sound_events,
        )
        d2 = engine2.run(inp_yamnet)

        assert d2.final_score >= d1.final_score

    def test_same_source_suppressed(self, engine, sample_features):
        """Two ALERT-level turns from the same source → second is suppressed."""
        threat1 = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.HIGH, threat_score=0.85,
            is_directed=True, reasoning="t", keywords_detected=["hurt"],
            confidence_in_direction=0.9,
        )
        threat2 = ThreatAssessment(
            turn_id=1, threat_level=ThreatLevel.HIGH, threat_score=0.85,
            is_directed=True, reasoning="t", keywords_detected=["hurt"],
            confidence_in_direction=0.9,
        )
        d1 = engine.run(AlertDecisionInput(threat1, sample_features, source_id="file_a", asr_confidence=0.9))
        d2 = engine.run(AlertDecisionInput(threat2, sample_features, source_id="file_a", asr_confidence=0.9))

        assert not d1.suppressed
        assert d2.suppressed
        assert d2.incident_id == d1.incident_id

    def test_different_source_not_suppressed(self, engine, sample_features):
        """Two ALERT-level turns from different sources → neither suppressed."""
        threat1 = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.HIGH, threat_score=0.85,
            is_directed=True, reasoning="t", keywords_detected=["hurt"],
            confidence_in_direction=0.9,
        )
        threat2 = ThreatAssessment(
            turn_id=1, threat_level=ThreatLevel.HIGH, threat_score=0.85,
            is_directed=True, reasoning="t", keywords_detected=["hurt"],
            confidence_in_direction=0.9,
        )
        d1 = engine.run(AlertDecisionInput(threat1, sample_features, source_id="file_a", asr_confidence=0.9))
        d2 = engine.run(AlertDecisionInput(threat2, sample_features, source_id="file_b", asr_confidence=0.9))

        assert not d1.suppressed
        assert not d2.suppressed
        assert d1.incident_id != d2.incident_id

    def test_cooldown_expiry(self, settings, sample_features):
        """After cooldown expires, same source is no longer suppressed."""
        settings.cooldown_seconds = 0.1  # very short cooldown
        engine = AlertDecisionEngine(settings)

        threat1 = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.HIGH, threat_score=0.85,
            is_directed=True, reasoning="t", keywords_detected=["hurt"],
            confidence_in_direction=0.9,
        )
        threat2 = ThreatAssessment(
            turn_id=1, threat_level=ThreatLevel.HIGH, threat_score=0.85,
            is_directed=True, reasoning="t", keywords_detected=["hurt"],
            confidence_in_direction=0.9,
        )
        engine.run(AlertDecisionInput(threat1, sample_features, source_id="file_a", asr_confidence=0.9))
        time.sleep(0.15)
        d2 = engine.run(AlertDecisionInput(threat2, sample_features, source_id="file_a", asr_confidence=0.9))

        assert not d2.suppressed
