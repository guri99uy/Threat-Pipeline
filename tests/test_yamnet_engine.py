"""Tests for the YAMNet engine (mocked model)."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from threat_pipeline.engines.yamnet_engine import YAMNetEngine
from threat_pipeline.models import SpeechTurn


class TestYAMNetEngineMocked:
    @pytest.fixture
    def engine(self, settings):
        engine = YAMNetEngine(settings)

        # Build class names (521 AudioSet classes, inject threat classes)
        class_names = [f"class_{i}" for i in range(521)]
        class_names[0] = "Speech"
        class_names[1] = "Gunshot, gunfire"
        class_names[2] = "Explosion"
        class_names[3] = "Glass"
        class_names[4] = "Screaming"
        engine._class_names = class_names

        # Mock model callable
        mock_model = MagicMock()
        # Scores: 10 frames × 521 classes
        scores = np.zeros((10, 521), dtype=np.float32)
        scores[:, 1] = 0.8  # Gunshot high
        scores[:, 4] = 0.6  # Screaming moderate

        mock_model.return_value = (
            MagicMock(numpy=MagicMock(return_value=scores)),
            MagicMock(),  # embeddings
            MagicMock(),  # spectrogram
        )
        engine._model = mock_model
        return engine

    def test_detects_threat_sounds(self, engine, sample_turn):
        result = engine.run(sample_turn)

        assert result.turn_id == 0
        assert result.has_threat_sound is True
        assert len(result.events) >= 1
        # Gunshot should be detected (score 0.8 > threshold 0.5)
        gunshot_events = [e for e in result.events if "Gunshot" in e["class"]]
        assert len(gunshot_events) == 1
        assert gunshot_events[0]["confidence"] == pytest.approx(0.8)

    def test_max_threat_confidence(self, engine, sample_turn):
        result = engine.run(sample_turn)
        assert result.max_threat_event_confidence == pytest.approx(0.8)

    def test_no_threat_sounds(self, settings, sample_turn):
        engine = YAMNetEngine(settings)
        engine._class_names = ["Speech", "Music", "Silence"]

        scores = np.zeros((10, 3), dtype=np.float32)
        scores[:, 0] = 0.9  # Just speech

        mock_model = MagicMock()
        mock_model.return_value = (
            MagicMock(numpy=MagicMock(return_value=scores)),
            MagicMock(),
            MagicMock(),
        )
        engine._model = mock_model

        result = engine.run(sample_turn)
        assert result.has_threat_sound is False
        assert len(result.events) == 0

    def test_latency_measured(self, engine, sample_turn):
        engine.run(sample_turn)
        assert engine.last_latency_s > 0
