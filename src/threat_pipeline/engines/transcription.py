"""Engine 3 — Speech-to-text via OpenAI Whisper API.

Converts each speech turn into text using the Whisper API.  A cleaning
step normalises whitespace and removes filler artefacts.

C#3 upgrade: Uses verbose_json response format to extract ASR confidence
(avg_logprob) and no_speech_prob for downstream fusion weighting.

Production note: for on-device / low-latency deployment, whisper.cpp or
a distilled ONNX Whisper model would replace this API call, eliminating
the network round-trip entirely.
"""

from __future__ import annotations

import io
import math
import re

import numpy as np
import soundfile as sf
import torch
from openai import OpenAI

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import SpeechTurn, TranscriptionResult


def _build_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        max_retries=settings.api_max_retries,
        timeout=settings.api_timeout_seconds,
    )


class TranscriptionEngine(Engine[SpeechTurn, TranscriptionResult]):
    """Transcribe a speech turn using the OpenAI Whisper API."""

    name = "transcription"

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self._client = client or _build_client(self.settings)

    def process(self, turn: SpeechTurn) -> TranscriptionResult:
        # Write turn audio to an in-memory WAV buffer using soundfile
        samples: torch.Tensor = turn.audio_samples
        buf = io.BytesIO()
        sf.write(buf, samples.numpy(), turn.sample_rate, format="WAV")
        buf.seek(0)
        buf.name = f"turn_{turn.turn_id}.wav"

        response = self._client.audio.transcriptions.create(
            model=self.settings.whisper_model,
            file=buf,
            response_format="verbose_json",
            language=self.settings.whisper_language,
        )

        # Parse verbose_json response
        raw_text, asr_confidence, no_speech_prob = self._parse_verbose_json(response)
        cleaned = self._clean(raw_text)

        # Empty transcript guard
        s = self.settings
        low_asr_confidence = False
        if not cleaned or no_speech_prob > s.no_speech_prob_threshold:
            low_asr_confidence = True
            asr_confidence = 0.0

        if asr_confidence < s.asr_confidence_threshold:
            low_asr_confidence = True

        return TranscriptionResult(
            turn_id=turn.turn_id,
            raw_text=raw_text,
            cleaned_text=cleaned,
            asr_confidence=asr_confidence,
            no_speech_prob=no_speech_prob,
            low_asr_confidence=low_asr_confidence,
        )

    @staticmethod
    def _parse_verbose_json(response) -> tuple[str, float, float]:
        """Extract text, confidence, and no_speech_prob from verbose_json response."""
        # The response object has .text, .segments, etc.
        text = ""
        avg_logprob = 0.0
        no_speech_prob = 0.0

        if hasattr(response, "text"):
            text = str(response.text).strip()
        elif isinstance(response, dict):
            text = str(response.get("text", "")).strip()

        # Extract avg_logprob and no_speech_prob from segments
        segments = None
        if hasattr(response, "segments"):
            segments = response.segments
        elif isinstance(response, dict):
            segments = response.get("segments", [])

        if segments:
            logprobs = []
            no_speech_probs = []
            for seg in segments:
                if hasattr(seg, "avg_logprob"):
                    lp = seg.avg_logprob
                    nsp = seg.no_speech_prob
                elif isinstance(seg, dict):
                    lp = seg.get("avg_logprob", 0.0)
                    nsp = seg.get("no_speech_prob", 0.0)
                else:
                    continue
                if lp is not None:
                    logprobs.append(lp)
                if nsp is not None:
                    no_speech_probs.append(nsp)

            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
            if no_speech_probs:
                no_speech_prob = max(no_speech_probs)

        # Convert avg_logprob to confidence via exp()
        asr_confidence = math.exp(avg_logprob) if avg_logprob < 0 else 1.0

        return text, asr_confidence, no_speech_prob

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and strip common ASR artefacts."""
        text = re.sub(r"\s+", " ", text).strip()
        # Remove leading/trailing punctuation artefacts
        text = text.strip(".-… ")
        return text
