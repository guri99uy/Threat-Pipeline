"""Tests for the TextClassifier engine (mocked model)."""

import pytest
from unittest.mock import MagicMock, patch

from threat_pipeline.engines.text_classifier_engine import TextClassifierEngine
from threat_pipeline.models import TranscriptionResult


class TestTextClassifierMocked:
    @pytest.fixture
    def engine(self, settings):
        engine = TextClassifierEngine(settings)
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "toxic", "score": 0.92}]
        engine._pipeline = mock_pipeline
        return engine

    def test_toxic_text(self, engine):
        transcript = TranscriptionResult(
            turn_id=0, raw_text="I will kill you",
            cleaned_text="I will kill you",
        )
        result = engine.run(transcript)

        assert result.turn_id == 0
        assert result.toxicity_score == pytest.approx(0.92)
        assert result.label == "toxic"
        assert result.confidence == pytest.approx(0.92)

    def test_neutral_text(self, settings):
        engine = TextClassifierEngine(settings)
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.95}]
        engine._pipeline = mock_pipeline

        transcript = TranscriptionResult(
            turn_id=0, raw_text="Hello how are you",
            cleaned_text="Hello how are you",
        )
        result = engine.run(transcript)

        assert result.toxicity_score == pytest.approx(0.05)
        assert result.label == "neutral"

    def test_empty_text(self, settings):
        engine = TextClassifierEngine(settings)
        engine._pipeline = MagicMock()  # shouldn't be called

        transcript = TranscriptionResult(
            turn_id=0, raw_text="", cleaned_text="",
        )
        result = engine.run(transcript)

        assert result.toxicity_score == 0.0
        assert result.label == "neutral"

    def test_latency_measured(self, engine):
        transcript = TranscriptionResult(
            turn_id=0, raw_text="test", cleaned_text="test",
        )
        engine.run(transcript)
        assert engine.last_latency_s > 0
