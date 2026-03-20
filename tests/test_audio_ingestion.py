"""Tests for the AudioIngestion engine."""

from pathlib import Path

import pytest
import torch

from threat_pipeline.engines.audio_ingestion import AudioIngestionEngine

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
WAV_FILES = sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []


class TestAudioIngestion:
    @pytest.fixture
    def engine(self, settings):
        return AudioIngestionEngine(settings)

    @pytest.mark.parametrize("wav", WAV_FILES, ids=[w.name for w in WAV_FILES])
    def test_load_real_wav(self, engine, wav):
        """Each WAV should load and resample to 16 kHz."""
        segment = engine.run(str(wav))

        assert segment.sample_rate == 16_000
        assert segment.duration_s > 0
        assert isinstance(segment.samples, torch.Tensor)
        assert segment.samples.dim() == 1  # mono, 1-D

    @pytest.mark.parametrize("wav", WAV_FILES, ids=[w.name for w in WAV_FILES])
    def test_latency_measured(self, engine, wav):
        engine.run(str(wav))
        assert engine.last_latency_s > 0

    def test_nonexistent_file_raises(self, engine):
        with pytest.raises(Exception):
            engine.run("nonexistent.wav")
