"""Tests for CLI JSON output mode."""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from threat_pipeline.cli import _serialize_result, _strip_tensors
from threat_pipeline.models import (
    AlertAction,
    AlertDecision,
    AudioFeatures,
    PipelineResult,
    SpeechTurn,
    ThreatAssessment,
    ThreatLevel,
    TranscriptionResult,
    TurnResult,
)


@pytest.fixture
def sample_pipeline_result():
    """A minimal PipelineResult with one turn for JSON tests."""
    turn = SpeechTurn(
        turn_id=0, start_s=0.0, end_s=0.5,
        audio_samples=torch.randn(8000), sample_rate=16000,
    )
    transcript = TranscriptionResult(
        turn_id=0, raw_text="test", cleaned_text="test",
        asr_confidence=0.8, no_speech_prob=0.05,
    )
    features = AudioFeatures(
        turn_id=0, rms_db=-20.0, spectral_centroid_mean=2000.0,
        zero_crossing_rate=0.05, is_loud=False, is_sharp=False,
        rms_normalized=0.5, centroid_normalized=0.3,
    )
    threat = ThreatAssessment(
        turn_id=0, threat_level=ThreatLevel.NONE, threat_score=0.1,
        is_directed=False, reasoning="benign", keywords_detected=[],
        confidence_in_direction=0.8,
    )
    decision = AlertDecision(
        turn_id=0, action=AlertAction.NO_ACTION, final_score=0.06,
        suppressed=False, incident_id="abc123",
    )
    turn_result = TurnResult(
        turn=turn, transcription=transcript, features=features,
        threat=threat, decision=decision,
    )
    return PipelineResult(
        source_path="test.wav", total_turns=1,
        turn_results=[turn_result], alerts_fired=0,
        engine_timings={"audio_ingestion": 0.01}, total_latency_s=0.01,
    )


class TestSerializeResult:
    def test_valid_json(self, sample_pipeline_result):
        """Serialized result should produce valid JSON."""
        data = _serialize_result(sample_pipeline_result)
        output = json.dumps(data)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_expected_keys(self, sample_pipeline_result):
        """Top-level keys should include pipeline metadata."""
        data = _serialize_result(sample_pipeline_result)
        assert "source_path" in data
        assert "total_turns" in data
        assert "turn_results" in data
        assert "alerts_fired" in data
        assert "engine_timings" in data
        assert "total_latency_s" in data

    def test_no_tensor_fields(self, sample_pipeline_result):
        """Serialized output should not contain any torch.Tensor objects."""
        data = _serialize_result(sample_pipeline_result)
        raw = json.dumps(data)
        # If a Tensor leaked through, json.dumps would raise TypeError
        # but also check the structure directly
        for tr in data["turn_results"]:
            turn_data = tr["turn"]
            assert "audio_samples" not in turn_data
            assert "samples" not in turn_data


class TestStripTensors:
    def test_removes_tensor_from_dict(self):
        d = {"a": 1, "b": torch.tensor([1.0]), "c": "hello"}
        _strip_tensors(d)
        assert "b" not in d
        assert d == {"a": 1, "c": "hello"}

    def test_removes_nested_tensor(self):
        d = {"outer": {"inner": torch.tensor([2.0]), "keep": True}}
        _strip_tensors(d)
        assert "inner" not in d["outer"]
        assert d["outer"]["keep"] is True
