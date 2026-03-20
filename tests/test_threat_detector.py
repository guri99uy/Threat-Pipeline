"""Tests for the ThreatDetector engine."""

import json

import pytest
from unittest.mock import MagicMock

from threat_pipeline.engines.threat_detector import ThreatDetectorEngine, ThreatDetectorInput
from threat_pipeline.models import ThreatLevel


class TestThreatDetectorMocked:
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

    def test_high_threat(self, settings, sample_turn, sample_transcription, sample_features):
        mock = self._make_mock_client({
            "threat_level": "high",
            "threat_score": 0.9,
            "is_directed": True,
            "confidence_in_direction": 0.85,
            "reasoning": "Direct physical threat",
            "keywords_detected": ["hurt"],
        })
        engine = ThreatDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(sample_turn, sample_transcription, sample_features)
        result = engine.run(inp)

        assert result.threat_level == ThreatLevel.HIGH
        assert result.threat_score == 0.9
        assert result.is_directed is True
        assert result.confidence_in_direction == 0.85
        assert "hurt" in result.keywords_detected

    def test_no_threat(self, settings, sample_turn, sample_features):
        from threat_pipeline.models import TranscriptionResult
        transcript = TranscriptionResult(
            turn_id=0,
            raw_text="Hey how are you doing today?",
            cleaned_text="Hey how are you doing today",
        )
        mock = self._make_mock_client({
            "threat_level": "none",
            "threat_score": 0.02,
            "is_directed": False,
            "confidence_in_direction": 0.9,
            "reasoning": "Casual greeting",
            "keywords_detected": [],
        })
        engine = ThreatDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(sample_turn, transcript, sample_features)
        result = engine.run(inp)

        assert result.threat_level == ThreatLevel.NONE
        assert result.threat_score < 0.1

    def test_score_clamped(self, settings, sample_turn, sample_transcription, sample_features):
        """Scores outside [0,1] should be clamped."""
        mock = self._make_mock_client({
            "threat_level": "critical",
            "threat_score": 1.5,
            "is_directed": True,
            "confidence_in_direction": 1.5,
            "reasoning": "test",
            "keywords_detected": [],
        })
        engine = ThreatDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(sample_turn, sample_transcription, sample_features)
        result = engine.run(inp)
        assert result.threat_score == 1.0
        assert result.confidence_in_direction == 1.0

    def test_prior_context_in_prompt(self, settings, sample_turn, sample_transcription, sample_features):
        """Prior context (tuples) should appear in the user message sent to the LLM."""
        mock = self._make_mock_client({
            "threat_level": "medium",
            "threat_score": 0.5,
            "is_directed": False,
            "confidence_in_direction": 0.6,
            "reasoning": "Escalating conversation",
            "keywords_detected": [],
        })
        engine = ThreatDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(
            sample_turn, sample_transcription, sample_features,
            prior_context=[
                ("Hey what do you want", 0.2, "low"),
                ("I said give me your phone", 0.6, "medium"),
            ],
        )
        engine.run(inp)

        call_args = mock.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "Previous turns" in user_msg
        assert "Hey what do you want" in user_msg
        assert "I said give me your phone" in user_msg

    def test_asr_confidence_in_prompt(self, settings, sample_turn, sample_transcription, sample_features):
        """ASR confidence should appear in the prompt."""
        mock = self._make_mock_client({
            "threat_level": "none",
            "threat_score": 0.1,
            "is_directed": False,
            "confidence_in_direction": 0.5,
            "reasoning": "test",
            "keywords_detected": [],
        })
        engine = ThreatDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(
            sample_turn, sample_transcription, sample_features,
            asr_confidence=0.42,
        )
        engine.run(inp)

        call_args = mock.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "ASR confidence: 0.42" in user_msg

    def test_lower_temperature(self, settings, sample_turn, sample_transcription, sample_features):
        """Temperature should use the configured value (0.05)."""
        mock = self._make_mock_client({
            "threat_level": "none",
            "threat_score": 0.1,
            "is_directed": False,
            "confidence_in_direction": 0.5,
            "reasoning": "test",
            "keywords_detected": [],
        })
        engine = ThreatDetectorEngine(settings, client=mock)
        inp = ThreatDetectorInput(sample_turn, sample_transcription, sample_features)
        engine.run(inp)

        call_args = mock.chat.completions.create.call_args
        assert call_args[1]["temperature"] == settings.threat_detector_temperature


@pytest.mark.api
class TestThreatDetectorAPI:
    """Integration tests requiring OpenAI API."""

    @pytest.fixture
    def engine(self):
        from threat_pipeline.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return ThreatDetectorEngine(settings)

    def test_real_threat_assessment(self, engine, sample_turn, sample_transcription, sample_features):
        inp = ThreatDetectorInput(sample_turn, sample_transcription, sample_features)
        result = engine.run(inp)

        assert result.threat_level in list(ThreatLevel)
        assert 0.0 <= result.threat_score <= 1.0
        assert 0.0 <= result.confidence_in_direction <= 1.0
        assert isinstance(result.reasoning, str)

    def test_figurative_speech_low_threat(self, engine, sample_turn, sample_features):
        """Figurative speech should not be flagged as high threat."""
        from threat_pipeline.models import TranscriptionResult
        transcript = TranscriptionResult(
            turn_id=0,
            raw_text="Oh my god I could kill for a coffee right now",
            cleaned_text="Oh my god I could kill for a coffee right now",
        )
        inp = ThreatDetectorInput(sample_turn, transcript, sample_features)
        result = engine.run(inp)

        assert result.threat_level in (ThreatLevel.NONE, ThreatLevel.LOW)
        assert result.threat_score < 0.4

    def test_sarcasm_low_threat(self, engine, sample_turn, sample_features):
        """Sarcastic remark should not be flagged as high threat."""
        from threat_pipeline.models import TranscriptionResult
        transcript = TranscriptionResult(
            turn_id=0,
            raw_text="Oh sure I'll just stab myself with this pen, great idea",
            cleaned_text="Oh sure I'll just stab myself with this pen great idea",
        )
        inp = ThreatDetectorInput(sample_turn, transcript, sample_features)
        result = engine.run(inp)

        assert result.threat_level in (ThreatLevel.NONE, ThreatLevel.LOW)
        assert result.threat_score < 0.4

    def test_self_directed_distress(self, engine, sample_turn, sample_features):
        """Self-directed distress is not a threat to others."""
        from threat_pipeline.models import TranscriptionResult
        transcript = TranscriptionResult(
            turn_id=0,
            raw_text="I just want to disappear, I hate everything about myself",
            cleaned_text="I just want to disappear I hate everything about myself",
        )
        inp = ThreatDetectorInput(sample_turn, transcript, sample_features)
        result = engine.run(inp)

        assert result.is_directed is False
        assert result.threat_score < 0.5
