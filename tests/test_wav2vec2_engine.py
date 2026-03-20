"""Tests for the Wav2Vec2 engine (mocked model)."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from threat_pipeline.engines.wav2vec2_engine import Wav2Vec2Engine
from threat_pipeline.models import SpeechTurn


class TestWav2Vec2EngineMocked:
    @pytest.fixture
    def engine(self, settings):
        engine = Wav2Vec2Engine(settings)
        # Mock model and processor
        mock_processor = MagicMock()
        mock_processor.return_value = MagicMock(
            input_values=torch.randn(1, 8000)
        )
        mock_processor.batch_decode.return_value = ["HELLO WORLD"]

        mock_model = MagicMock()
        mock_logits = torch.randn(1, 50, 32)
        mock_hidden = torch.randn(1, 50, 768)
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_output.hidden_states = [mock_hidden, mock_hidden]  # 2 layers
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()

        engine._processor = mock_processor
        engine._model = mock_model
        return engine

    def test_returns_ctc_transcript(self, engine, sample_turn):
        result = engine.run(sample_turn)
        assert result.turn_id == 0
        assert result.ctc_transcript == "HELLO WORLD"

    def test_returns_embedding(self, engine, sample_turn):
        result = engine.run(sample_turn)
        assert len(result.embedding) == 768
        assert all(isinstance(v, float) for v in result.embedding)

    def test_latency_measured(self, engine, sample_turn):
        engine.run(sample_turn)
        assert engine.last_latency_s > 0
