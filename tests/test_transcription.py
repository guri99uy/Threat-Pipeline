"""Tests for the Transcription engine."""

import pytest
from unittest.mock import MagicMock, patch

from threat_pipeline.engines.transcription import TranscriptionEngine


class TestTranscriptionCleaning:
    def test_clean_whitespace(self):
        assert TranscriptionEngine._clean("  hello   world  ") == "hello world"

    def test_clean_artefacts(self):
        assert TranscriptionEngine._clean("...hello world...") == "hello world"

    def test_clean_empty(self):
        assert TranscriptionEngine._clean("") == ""


class TestParseVerboseJson:
    def test_parse_dict_response(self):
        response = {
            "text": " Hello world. ",
            "segments": [
                {"avg_logprob": -0.3, "no_speech_prob": 0.05},
                {"avg_logprob": -0.4, "no_speech_prob": 0.02},
            ],
        }
        text, conf, nsp = TranscriptionEngine._parse_verbose_json(response)
        assert text == "Hello world."
        assert 0.0 < conf < 1.0
        assert nsp == 0.05

    def test_parse_object_response(self):
        response = MagicMock()
        response.text = "Hi there"
        seg1 = MagicMock()
        seg1.avg_logprob = -0.2
        seg1.no_speech_prob = 0.01
        response.segments = [seg1]

        text, conf, nsp = TranscriptionEngine._parse_verbose_json(response)
        assert text == "Hi there"
        assert 0.0 < conf < 1.0
        assert nsp == 0.01

    def test_parse_empty_segments(self):
        response = {"text": "test", "segments": []}
        text, conf, nsp = TranscriptionEngine._parse_verbose_json(response)
        assert text == "test"
        assert conf == 1.0  # no logprobs → default 1.0
        assert nsp == 0.0


class TestTranscriptionMocked:
    def test_transcribe_returns_result_with_confidence(self, settings, sample_turn):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello world."
        seg = MagicMock()
        seg.avg_logprob = -0.25
        seg.no_speech_prob = 0.03
        mock_response.segments = [seg]
        mock_client.audio.transcriptions.create.return_value = mock_response

        engine = TranscriptionEngine(settings, client=mock_client)
        result = engine.run(sample_turn)

        assert result.turn_id == 0
        assert result.raw_text == "Hello world."
        assert result.cleaned_text == "Hello world"
        assert 0.0 < result.asr_confidence < 1.0
        assert result.no_speech_prob == 0.03
        assert result.low_asr_confidence is False
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["language"] == settings.whisper_language

    def test_empty_transcript_low_confidence(self, settings, sample_turn):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "   "
        mock_response.segments = []
        mock_client.audio.transcriptions.create.return_value = mock_response

        engine = TranscriptionEngine(settings, client=mock_client)
        result = engine.run(sample_turn)

        assert result.cleaned_text == ""
        assert result.low_asr_confidence is True
        assert result.asr_confidence == 0.0

    def test_high_no_speech_prob_flags_low_confidence(self, settings, sample_turn):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "uh"
        seg = MagicMock()
        seg.avg_logprob = -0.5
        seg.no_speech_prob = 0.8  # above threshold
        mock_response.segments = [seg]
        mock_client.audio.transcriptions.create.return_value = mock_response

        engine = TranscriptionEngine(settings, client=mock_client)
        result = engine.run(sample_turn)

        assert result.low_asr_confidence is True
        assert result.asr_confidence == 0.0


@pytest.mark.api
class TestTranscriptionAPI:
    """These tests require a real OpenAI API key."""

    @pytest.fixture
    def engine(self):
        from threat_pipeline.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return TranscriptionEngine(settings)

    def test_transcribe_real_audio(self, engine, sample_turn):
        result = engine.run(sample_turn)
        assert result.turn_id == 0
        assert isinstance(result.raw_text, str)
        assert isinstance(result.cleaned_text, str)
        assert 0.0 <= result.asr_confidence <= 1.0
        assert 0.0 <= result.no_speech_prob <= 1.0
