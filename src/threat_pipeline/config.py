"""Centralised configuration loaded from environment / .env file."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    whisper_model: str = "whisper-1"
    # TUNABLE — enforce language for accented speakers
    whisper_language: str = "en"
    threat_model: str = "gpt-4o-mini"

    # Audio
    target_sample_rate: int = 16_000  # Silero VAD expects 16 kHz

    # VAD
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 100

    # Alert decision
    weight_threat: float = 0.6
    weight_loudness: float = 0.2
    weight_keyword: float = 0.2
    cooldown_seconds: float = 30.0

    # Context
    context_window_size: int = 2

    # Thresholds
    loud_rms_db: float = -20.0
    sharp_centroid_hz: float = 3000.0

    # ── C#1: IncidentState ──────────────────────────────────────────────
    # TUNABLE — C#1 — calibration candidate
    incident_ema_alpha: float = 0.4
    # TUNABLE — C#1 — calibration candidate
    escalation_consecutive_threshold: int = 3

    # ── C#2: Fusion & thresholds ────────────────────────────────────────
    # TUNABLE — C#2 — calibration candidate
    threshold_escalate: float = 0.9
    # TUNABLE — C#2 — calibration candidate
    threshold_alert: float = 0.7
    # TUNABLE — C#2 — calibration candidate
    threshold_log: float = 0.4
    # TUNABLE — C#2 — calibration candidate
    uncertain_score_low: float = 0.6
    # TUNABLE — C#2 — calibration candidate
    uncertain_score_high: float = 0.75
    # TUNABLE — C#2 — calibration candidate
    directed_boost: float = 0.15
    # TUNABLE — C#2 — calibration candidate (set after benchmark comparison)
    weight_text_classifier: float = 0.0
    # TUNABLE — C#9 — calibration candidate
    yamnet_weight_in_fusion: float = 0.1
    # TUNABLE — C#2 — calibration candidate
    speculative_llm_enabled: bool = True
    # TUNABLE — C#2 — calibration candidate
    speculative_llm_temperature: float = 0.7

    # ── API resilience (W-18) ────────────────────────────────────────────
    api_max_retries: int = 3
    api_timeout_seconds: float = 30.0

    # ── C#3: ASR confidence ─────────────────────────────────────────────
    # TUNABLE — C#3 — calibration candidate
    asr_confidence_threshold: float = 0.3
    # TUNABLE — C#3 — calibration candidate
    no_speech_prob_threshold: float = 0.5

    # ── C#4: Threat detector ────────────────────────────────────────────
    # TUNABLE — C#4 — calibration candidate
    threat_detector_temperature: float = 0.05

    # ── C#5: Audio features ─────────────────────────────────────────────
    # TUNABLE — C#5 — calibration candidate
    n_mfcc: int = 13

    # ── C#9: YAMNet ─────────────────────────────────────────────────────
    # TUNABLE — C#9 — calibration candidate
    yamnet_confidence_threshold: float = 0.5
    # TUNABLE — C#9 — calibration candidate
    yamnet_threat_classes: list[str] = [
        "Gunshot, gunfire",
        "Explosion",
        "Glass",
        "Screaming",
    ]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    return Settings()
