"""Tests for the VAD engine (Silero VAD)."""

from pathlib import Path

import pytest
import torch

from threat_pipeline.engines.audio_ingestion import AudioIngestionEngine
from threat_pipeline.engines.vad import VADEngine

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
WAV_FILES = sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []


class TestVAD:
    @pytest.fixture
    def ingest(self, settings):
        return AudioIngestionEngine(settings)

    @pytest.fixture
    def vad(self, settings):
        return VADEngine(settings)

    @pytest.mark.parametrize("wav", WAV_FILES, ids=[w.name for w in WAV_FILES])
    def test_detects_speech(self, ingest, vad, wav):
        """Each test WAV should produce at least one speech turn."""
        segment = ingest.run(str(wav))
        turns = vad.run(segment)

        assert len(turns) >= 1
        for turn in turns:
            assert turn.end_s > turn.start_s
            assert turn.sample_rate == 16_000
            assert isinstance(turn.audio_samples, torch.Tensor)
            assert turn.audio_samples.numel() > 0

    def test_silence_produces_no_turns(self, vad, settings):
        """Pure silence should yield zero turns."""
        from threat_pipeline.models import AudioSegment

        silence = AudioSegment(
            samples=torch.zeros(16_000 * 2),  # 2 seconds of silence
            sample_rate=16_000,
            duration_s=2.0,
            source_path="silence.wav",
        )
        turns = vad.run(silence)
        assert len(turns) == 0

    @pytest.mark.parametrize("wav", WAV_FILES[:1], ids=[w.name for w in WAV_FILES[:1]])
    def test_latency_measured(self, ingest, vad, wav):
        segment = ingest.run(str(wav))
        vad.run(segment)
        assert vad.last_latency_s > 0
