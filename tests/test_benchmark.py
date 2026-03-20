"""Tests for the benchmark harness."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from threat_pipeline.benchmark import _extract_run_metrics, RunMetrics, FileMetrics
from threat_pipeline.models import (
    AlertAction,
    AlertDecision,
    AudioFeatures,
    PipelineResult,
    SoundEventResult,
    TextClassification,
    ThreatAssessment,
    ThreatLevel,
    TranscriptionResult,
    TurnResult,
    SpeechTurn,
    Wav2Vec2Result,
)

import torch


class TestExtractRunMetrics:
    def test_extracts_from_pipeline_result(self):
        turn = SpeechTurn(
            turn_id=0, start_s=0.0, end_s=0.5,
            audio_samples=torch.randn(8000), sample_rate=16000,
        )
        result = PipelineResult(
            source_path="test.wav",
            total_turns=1,
            turn_results=[
                TurnResult(
                    turn=turn,
                    transcription=TranscriptionResult(
                        turn_id=0, raw_text="test", cleaned_text="test",
                        asr_confidence=0.8,
                    ),
                    threat=ThreatAssessment(
                        turn_id=0, threat_level=ThreatLevel.HIGH,
                        threat_score=0.85, is_directed=True,
                        reasoning="test", keywords_detected=["hurt"],
                    ),
                    decision=AlertDecision(
                        turn_id=0, action=AlertAction.ALERT, final_score=0.9,
                    ),
                    text_classification=TextClassification(
                        turn_id=0, toxicity_score=0.7, label="toxic", confidence=0.7,
                    ),
                    sound_events=SoundEventResult(
                        turn_id=0, events=[], max_threat_event_confidence=0.0,
                        has_threat_sound=False,
                    ),
                    wav2vec2_result=Wav2Vec2Result(
                        turn_id=0, ctc_transcript="TEST",
                    ),
                ),
            ],
            engine_timings={"transcription": 0.5},
            total_latency_s=1.0,
        )

        metrics = _extract_run_metrics("test.wav", result)
        assert metrics.file_name == "test.wav"
        assert metrics.final_score == 0.9
        assert metrics.threat_score == 0.85
        assert metrics.text_classifier_toxicity == 0.7
        assert metrics.wav2vec2_transcript == "TEST"
        assert metrics.asr_confidence == 0.8


class TestFileMetrics:
    def test_aggregation_properties(self):
        fm = FileMetrics(file_name="test.wav")
        fm.runs.append(RunMetrics(
            file_name="test.wav", final_score=0.8, action="alert",
            total_latency_s=1.0, engine_latencies={},
            threat_score=0.7, text_classifier_toxicity=0.5,
            yamnet_max_confidence=0.0, wav2vec2_transcript="",
            asr_confidence=0.9,
        ))
        fm.runs.append(RunMetrics(
            file_name="test.wav", final_score=0.9, action="escalate",
            total_latency_s=1.2, engine_latencies={},
            threat_score=0.8, text_classifier_toxicity=0.6,
            yamnet_max_confidence=0.0, wav2vec2_transcript="",
            asr_confidence=0.85,
        ))

        assert len(fm.final_scores) == 2
        assert fm.final_scores == [0.8, 0.9]
        assert fm.latencies == [1.0, 1.2]
        assert fm.threat_scores == [0.7, 0.8]
        assert fm.toxicity_scores == [0.5, 0.6]
