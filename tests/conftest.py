"""Shared fixtures for the threat pipeline test suite."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from threat_pipeline.config import Settings
from threat_pipeline.models import (
    AlertAction,
    AlertDecision,
    AudioFeatures,
    AudioSegment,
    IncidentSnapshot,
    SoundEventResult,
    SpeechTurn,
    TextClassification,
    ThreatAssessment,
    ThreatLevel,
    TranscriptionResult,
    Wav2Vec2Result,
)

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
WAV_FILES = sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []


@pytest.fixture
def audio_dir() -> Path:
    return AUDIO_DIR


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key")


@pytest.fixture
def sample_segment() -> AudioSegment:
    """1-second 16 kHz sine wave at 440 Hz."""
    sr = 16_000
    t = torch.linspace(0, 1, sr)
    samples = torch.sin(2 * torch.pi * 440 * t)
    return AudioSegment(
        samples=samples, sample_rate=sr, duration_s=1.0, source_path="test.wav"
    )


@pytest.fixture
def sample_turn() -> SpeechTurn:
    """Short speech turn (0.5s of white noise)."""
    sr = 16_000
    samples = torch.randn(sr // 2) * 0.1
    return SpeechTurn(
        turn_id=0, start_s=0.0, end_s=0.5, audio_samples=samples, sample_rate=sr
    )


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        turn_id=0,
        raw_text="Give me that phone or I will hurt you.",
        cleaned_text="Give me that phone or I will hurt you",
        asr_confidence=0.85,
        no_speech_prob=0.05,
        low_asr_confidence=False,
    )


@pytest.fixture
def sample_features() -> AudioFeatures:
    return AudioFeatures(
        turn_id=0,
        rms_db=-15.0,
        spectral_centroid_mean=3500.0,
        zero_crossing_rate=0.08,
        is_loud=True,
        is_sharp=True,
        mfcc_means=[1.0] * 13,
        spectral_contrast_mean=[10.0] * 7,
        mel_spectrogram_mean=-20.0,
        mel_spectrogram_std=5.0,
        spectral_bandwidth_mean=2000.0,
        rms_normalized=0.85,
        centroid_normalized=0.73,
    )


@pytest.fixture
def benign_features() -> AudioFeatures:
    return AudioFeatures(
        turn_id=0,
        rms_db=-30.0,
        spectral_centroid_mean=1500.0,
        zero_crossing_rate=0.04,
        is_loud=False,
        is_sharp=False,
        mfcc_means=[0.5] * 13,
        spectral_contrast_mean=[5.0] * 7,
        mel_spectrogram_mean=-40.0,
        mel_spectrogram_std=3.0,
        spectral_bandwidth_mean=1000.0,
        rms_normalized=0.15,
        centroid_normalized=0.27,
    )


@pytest.fixture
def sample_threat_high() -> ThreatAssessment:
    return ThreatAssessment(
        turn_id=0,
        threat_level=ThreatLevel.HIGH,
        threat_score=0.85,
        is_directed=True,
        reasoning="Direct physical threat with explicit intent to harm",
        keywords_detected=["hurt"],
        confidence_in_direction=0.9,
    )


@pytest.fixture
def sample_threat_none() -> ThreatAssessment:
    return ThreatAssessment(
        turn_id=0,
        threat_level=ThreatLevel.NONE,
        threat_score=0.05,
        is_directed=False,
        reasoning="Casual friendly conversation with no threat indicators",
        keywords_detected=[],
        confidence_in_direction=0.8,
    )
