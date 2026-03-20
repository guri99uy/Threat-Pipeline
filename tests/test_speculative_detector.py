"""Tests for the SpeculativeDetector engine."""

import json

import pytest
from unittest.mock import MagicMock

from threat_pipeline.engines.speculative_detector import SpeculativeDetectorEngine
from threat_pipeline.engines.threat_detector import ThreatDetectorInput
from threat_pipeline.models import ThreatLevel


class TestSpeculativeDetectorMocked:
    def _make_mock_client(self, response_dict: dict) -> MagicMock:
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps(response_dict)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        return mock_client

    def test_uses_speculative_temperature(self, settings, sample_turn, sample_transcription, sample_features):
        mock = self._make_mock_client({
            "threat_level": "medium",
            "threat_score": 0.6,
            "is_directed": False,
            "confidence_in_direction": 0.5,
            "reasoning": "Borderline case",
            "keywords_detected": [],
        })
        engine = SpeculativeDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(sample_turn, sample_transcription, sample_features)
        engine.run(inp)

        call_args = mock.chat.completions.create.call_args
        assert call_args[1]["temperature"] == settings.speculative_llm_temperature

    def test_returns_threat_assessment(self, settings, sample_turn, sample_transcription, sample_features):
        mock = self._make_mock_client({
            "threat_level": "high",
            "threat_score": 0.8,
            "is_directed": True,
            "confidence_in_direction": 0.7,
            "reasoning": "Elevated concern",
            "keywords_detected": ["hurt"],
        })
        engine = SpeculativeDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(sample_turn, sample_transcription, sample_features)
        result = engine.run(inp)

        assert result.threat_level == ThreatLevel.HIGH
        assert result.threat_score == 0.8
        assert result.confidence_in_direction == 0.7

    def test_prior_context_in_prompt(self, settings, sample_turn, sample_transcription, sample_features):
        mock = self._make_mock_client({
            "threat_level": "none",
            "threat_score": 0.1,
            "is_directed": False,
            "confidence_in_direction": 0.5,
            "reasoning": "test",
            "keywords_detected": [],
        })
        engine = SpeculativeDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(
            sample_turn, sample_transcription, sample_features,
            prior_context=[("previous text", 0.3, "low")],
        )
        engine.run(inp)

        call_args = mock.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "previous text" in user_msg
