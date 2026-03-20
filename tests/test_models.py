"""Unit tests for Pydantic data models."""

import torch
import pytest

from threat_pipeline.models import (
    AlertAction,
    AlertDecision,
    AlertEvent,
    AudioFeatures,
    AudioSegment,
    IncidentSnapshot,
    SoundEventResult,
    SpeechTurn,
    TextClassification,
    ThreatAssessment,
    ThreatLevel,
    TranscriptionResult,
    TurnResult,
    PipelineResult,
    Wav2Vec2Result,
)


class TestAudioSegment:
    def test_creation(self):
        seg = AudioSegment(
            samples=torch.zeros(16_000),
            sample_rate=16_000,
            duration_s=1.0,
            source_path="test.wav",
        )
        assert seg.sample_rate == 16_000
        assert seg.duration_s == 1.0

    def test_arbitrary_types(self):
        """Torch tensors should be accepted."""
        seg = AudioSegment(
            samples=torch.randn(8000),
            sample_rate=16_000,
            duration_s=0.5,
            source_path="x.wav",
        )
        assert isinstance(seg.samples, torch.Tensor)


class TestSpeechTurn:
    def test_duration_property(self):
        turn = SpeechTurn(
            turn_id=0,
            start_s=1.0,
            end_s=3.5,
            audio_samples=torch.zeros(40_000),
            sample_rate=16_000,
        )
        assert turn.duration_s == pytest.approx(2.5)


class TestTranscriptionResult:
    def test_asr_confidence_fields(self):
        tr = TranscriptionResult(
            turn_id=0,
            raw_text="hello",
            cleaned_text="hello",
            asr_confidence=0.75,
            no_speech_prob=0.1,
            low_asr_confidence=False,
        )
        assert tr.asr_confidence == 0.75
        assert tr.no_speech_prob == 0.1
        assert tr.low_asr_confidence is False

    def test_defaults(self):
        tr = TranscriptionResult(turn_id=0, raw_text="hi", cleaned_text="hi")
        assert tr.asr_confidence == 1.0
        assert tr.no_speech_prob == 0.0
        assert tr.low_asr_confidence is False


class TestAudioFeatures:
    def test_expanded_features(self):
        af = AudioFeatures(
            turn_id=0, rms_db=-20.0, spectral_centroid_mean=2000.0,
            zero_crossing_rate=0.05, is_loud=False, is_sharp=False,
            mfcc_means=[1.0] * 13,
            spectral_contrast_mean=[10.0] * 7,
            mel_spectrogram_mean=-30.0,
            mel_spectrogram_std=4.0,
            spectral_bandwidth_mean=1500.0,
            rms_normalized=0.5,
            centroid_normalized=0.3,
        )
        assert len(af.mfcc_means) == 13
        assert len(af.spectral_contrast_mean) == 7
        assert af.rms_normalized == 0.5


class TestThreatAssessment:
    def test_score_bounds(self):
        with pytest.raises(Exception):
            ThreatAssessment(
                turn_id=0,
                threat_level=ThreatLevel.HIGH,
                threat_score=1.5,  # out of bounds
                is_directed=True,
                reasoning="test",
                keywords_detected=[],
            )

    def test_valid(self):
        ta = ThreatAssessment(
            turn_id=0,
            threat_level=ThreatLevel.MEDIUM,
            threat_score=0.6,
            is_directed=False,
            reasoning="Elevated tone",
            keywords_detected=["angry"],
            confidence_in_direction=0.7,
        )
        assert ta.threat_score == 0.6
        assert ta.confidence_in_direction == 0.7

    def test_confidence_in_direction_default(self):
        ta = ThreatAssessment(
            turn_id=0, threat_level=ThreatLevel.NONE, threat_score=0.1,
            is_directed=False, reasoning="test", keywords_detected=[],
        )
        assert ta.confidence_in_direction == 0.5


class TestAlertAction:
    def test_uncertain_state(self):
        assert AlertAction.UNCERTAIN.value == "uncertain"
        assert AlertAction.UNCERTAIN in list(AlertAction)


class TestAlertDecision:
    def test_default_incident_id(self):
        d = AlertDecision(
            turn_id=0, action=AlertAction.LOG, final_score=0.5
        )
        assert len(d.incident_id) == 12

    def test_suppressed_default_false(self):
        d = AlertDecision(
            turn_id=0, action=AlertAction.NO_ACTION, final_score=0.1
        )
        assert d.suppressed is False


class TestAlertEvent:
    def test_full_event(self):
        evt = AlertEvent(
            incident_id="abc123",
            turn_id=0,
            action=AlertAction.ESCALATE,
            final_score=0.95,
            transcript="Give me that phone or I will hurt you",
            reasoning="Direct threat",
            keywords=["hurt"],
            audio_indicators={"rms_db": -15.0, "is_loud": True},
            latency_breakdown={"transcription": 0.5},
        )
        assert evt.action == AlertAction.ESCALATE


class TestWav2Vec2Result:
    def test_creation(self):
        r = Wav2Vec2Result(
            turn_id=0, ctc_transcript="hello world",
            embedding=[0.1] * 768,
        )
        assert len(r.embedding) == 768


class TestTextClassification:
    def test_creation(self):
        tc = TextClassification(
            turn_id=0, toxicity_score=0.85, label="toxic", confidence=0.9,
        )
        assert tc.toxicity_score == 0.85


class TestSoundEventResult:
    def test_creation(self):
        se = SoundEventResult(
            turn_id=0,
            events=[{"class": "Gunshot", "confidence": 0.9}],
            max_threat_event_confidence=0.9,
            has_threat_sound=True,
        )
        assert se.has_threat_sound is True


class TestIncidentSnapshot:
    def test_creation(self):
        snap = IncidentSnapshot(
            source_id="test", accumulated_score=0.5,
            consecutive_rising=2, turn_count=5, is_escalating=False,
        )
        assert snap.accumulated_score == 0.5


class TestPipelineResult:
    def test_defaults(self):
        pr = PipelineResult(source_path="test.wav", total_turns=0)
        assert pr.alerts_fired == 0
        assert pr.turn_results == []
