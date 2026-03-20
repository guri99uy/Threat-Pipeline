"""Tests for the AudioFeatures engine."""

import pytest
import torch

from threat_pipeline.engines.audio_features import AudioFeaturesEngine
from threat_pipeline.models import SpeechTurn


class TestAudioFeatures:
    @pytest.fixture
    def engine(self, settings):
        return AudioFeaturesEngine(settings)

    def test_extracts_features(self, engine, sample_turn):
        features = engine.run(sample_turn)

        assert features.turn_id == 0
        assert isinstance(features.rms_db, float)
        assert isinstance(features.spectral_centroid_mean, float)
        assert isinstance(features.zero_crossing_rate, float)
        assert isinstance(features.is_loud, bool)
        assert isinstance(features.is_sharp, bool)

    def test_expanded_features(self, engine, sample_turn):
        """C#5: expanded features should be populated."""
        features = engine.run(sample_turn)

        # MFCCs
        assert len(features.mfcc_means) == 13  # n_mfcc default
        assert all(isinstance(v, float) for v in features.mfcc_means)

        # Spectral contrast (7 bands)
        assert len(features.spectral_contrast_mean) == 7
        assert all(isinstance(v, float) for v in features.spectral_contrast_mean)

        # Mel spectrogram stats
        assert isinstance(features.mel_spectrogram_mean, float)
        assert isinstance(features.mel_spectrogram_std, float)

        # Spectral bandwidth
        assert isinstance(features.spectral_bandwidth_mean, float)

        # Continuous normalized values
        assert 0.0 <= features.rms_normalized <= 1.0
        assert 0.0 <= features.centroid_normalized <= 1.0

    def test_loud_signal_detected(self, engine):
        """A loud signal should be flagged."""
        sr = 16_000
        # Loud sine wave (amplitude 0.9)
        t = torch.linspace(0, 0.5, sr // 2)
        samples = 0.9 * torch.sin(2 * torch.pi * 440 * t)
        turn = SpeechTurn(
            turn_id=0, start_s=0.0, end_s=0.5, audio_samples=samples, sample_rate=sr
        )
        features = engine.run(turn)
        assert features.is_loud is True
        assert features.rms_normalized > 0.5

    def test_quiet_signal(self, engine):
        """A very quiet signal should not be flagged as loud."""
        sr = 16_000
        samples = torch.randn(sr // 2) * 0.001
        turn = SpeechTurn(
            turn_id=0, start_s=0.0, end_s=0.5, audio_samples=samples, sample_rate=sr
        )
        features = engine.run(turn)
        assert features.is_loud is False
        assert features.rms_normalized < 0.5

    def test_latency_measured(self, engine, sample_turn):
        engine.run(sample_turn)
        assert engine.last_latency_s > 0
