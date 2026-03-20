"""Pydantic I/O contracts for every engine in the threat detection pipeline."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

import torch
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Engine 1 — AudioIngest
# ---------------------------------------------------------------------------

class AudioSegment(BaseModel):
    """Raw waveform loaded and resampled to a canonical sample rate."""

    samples: object  # torch.Tensor — kept as object for Pydantic compat
    sample_rate: int
    duration_s: float
    source_path: str

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Engine 2 — VAD
# ---------------------------------------------------------------------------

class SpeechTurn(BaseModel):
    """A single speech segment extracted by VAD."""

    turn_id: int
    start_s: float
    end_s: float
    audio_samples: object  # torch.Tensor
    sample_rate: int

    model_config = {"arbitrary_types_allowed": True}

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Engine 3 — Transcription (C#3: ASR confidence)
# ---------------------------------------------------------------------------

class TranscriptionResult(BaseModel):
    """Whisper transcription of a single speech turn."""

    turn_id: int
    raw_text: str
    cleaned_text: str
    asr_confidence: float = 1.0
    no_speech_prob: float = 0.0
    low_asr_confidence: bool = False


# ---------------------------------------------------------------------------
# Engine 4 — Audio Features (C#5: expanded)
# ---------------------------------------------------------------------------

class AudioFeatures(BaseModel):
    """Deterministic audio-level signals extracted via librosa."""

    turn_id: int
    rms_db: float
    spectral_centroid_mean: float
    zero_crossing_rate: float
    is_loud: bool
    is_sharp: bool
    # C#5: Expanded features
    mfcc_means: list[float] = Field(default_factory=list)
    spectral_contrast_mean: list[float] = Field(default_factory=list)
    mel_spectrogram_mean: float = 0.0
    mel_spectrogram_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    rms_normalized: float = 0.0
    centroid_normalized: float = 0.0


# ---------------------------------------------------------------------------
# Engine 5 — Threat Detection (LLM) — C#4: confidence_in_direction
# ---------------------------------------------------------------------------

class ThreatLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatAssessment(BaseModel):
    """Structured output from GPT-4o-mini threat reasoning."""

    turn_id: int
    threat_level: ThreatLevel
    threat_score: float = Field(ge=0.0, le=1.0)
    is_directed: bool
    reasoning: str
    keywords_detected: list[str] = Field(default_factory=list)
    confidence_in_direction: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Engine 6 — Alert Decision (C#2: UNCERTAIN state)
# ---------------------------------------------------------------------------

class AlertAction(str, Enum):
    NO_ACTION = "no_action"
    LOG = "log"
    UNCERTAIN = "uncertain"
    ALERT = "alert"
    ESCALATE = "escalate"


class AlertDecision(BaseModel):
    """Fusion result combining LLM + audio + heuristic signals."""

    turn_id: int
    action: AlertAction
    final_score: float = Field(ge=0.0, le=1.0)
    suppressed: bool = False
    incident_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])


# ---------------------------------------------------------------------------
# Engine 7 — Alert Publisher
# ---------------------------------------------------------------------------

class AlertEvent(BaseModel):
    """Full-context alert published to EventBus / downstream consumers."""

    incident_id: str
    turn_id: int
    action: AlertAction
    final_score: float
    transcript: str
    reasoning: str
    keywords: list[str]
    audio_indicators: dict
    latency_breakdown: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# C#8: Wav2Vec2 result
# ---------------------------------------------------------------------------

class Wav2Vec2Result(BaseModel):
    """Dual output from wav2vec2: CTC transcript + embedding."""

    turn_id: int
    ctc_transcript: str
    embedding: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# C#7: Text classification (DistilRoBERTa toxicity)
# ---------------------------------------------------------------------------

class TextClassification(BaseModel):
    """Pre-trained toxicity classifier output."""

    turn_id: int
    toxicity_score: float
    label: str
    confidence: float


# ---------------------------------------------------------------------------
# C#9: YAMNet sound events
# ---------------------------------------------------------------------------

class SoundEventResult(BaseModel):
    """YAMNet sound event detection output."""

    turn_id: int
    events: list[dict] = Field(default_factory=list)
    max_threat_event_confidence: float = 0.0
    has_threat_sound: bool = False


# ---------------------------------------------------------------------------
# C#1: IncidentState snapshot
# ---------------------------------------------------------------------------

class IncidentSnapshot(BaseModel):
    """Per-source incident tracking state."""

    source_id: str
    accumulated_score: float
    consecutive_rising: int
    turn_count: int
    is_escalating: bool


# ---------------------------------------------------------------------------
# Pipeline-level
# ---------------------------------------------------------------------------

class TurnResult(BaseModel):
    """All engine outputs for a single speech turn."""

    turn: SpeechTurn
    transcription: Optional[TranscriptionResult] = None
    features: Optional[AudioFeatures] = None
    threat: Optional[ThreatAssessment] = None
    decision: Optional[AlertDecision] = None
    alert: Optional[AlertEvent] = None
    wav2vec2_result: Optional[Wav2Vec2Result] = None
    text_classification: Optional[TextClassification] = None
    sound_events: Optional[SoundEventResult] = None
    incident_snapshot: Optional[IncidentSnapshot] = None

    model_config = {"arbitrary_types_allowed": True}


class PipelineResult(BaseModel):
    """Aggregated result of processing a full audio file."""

    source_path: str
    total_turns: int
    turn_results: list[TurnResult] = Field(default_factory=list)
    alerts_fired: int = 0
    engine_timings: dict[str, float] = Field(default_factory=dict)
    total_latency_s: float = 0.0

    model_config = {"arbitrary_types_allowed": True}
