"""Engine 5 — LLM-based threat reasoning with Structured Outputs.

Uses GPT-4o-mini with a JSON schema to produce an explainable
ThreatAssessment for each speech turn.

C#3 upgrade: ASR confidence passed in prompt, weights assessment.
C#4 upgrade: confidence_in_direction field, concise prompt, lower temperature.

Production note: this is the most latency-expensive step (~300-800 ms).
For edge deployment a fine-tuned, quantised classifier (ONNX) would
replace the API call.
"""

from __future__ import annotations

import json

from openai import OpenAI

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import (
    AudioFeatures,
    SpeechTurn,
    ThreatAssessment,
    ThreatLevel,
    TranscriptionResult,
)


def _build_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        max_retries=settings.api_max_retries,
        timeout=settings.api_timeout_seconds,
    )


class ThreatDetectorInput:
    def __init__(
        self,
        turn: SpeechTurn,
        transcription: TranscriptionResult,
        features: AudioFeatures,
        prior_context: list[tuple[str, float, str]] | None = None,
        asr_confidence: float = 1.0,
    ):
        self.turn = turn
        self.transcription = transcription
        self.features = features
        self.prior_context: list[tuple[str, float, str]] = prior_context or []
        self.asr_confidence = asr_confidence


# JSON schema for OpenAI Structured Outputs — C#4: added confidence_in_direction
_THREAT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "threat_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "threat_level": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "critical"],
                },
                "threat_score": {
                    "type": "number",
                    "description": "0.0 (safe) to 1.0 (maximum threat)",
                },
                "is_directed": {
                    "type": "boolean",
                    "description": "True if the threat is directed at a specific person present",
                },
                "confidence_in_direction": {
                    "type": "number",
                    "description": "0.0-1.0: how confident you are in the is_directed assessment",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the assessment",
                },
                "keywords_detected": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Threat-related keywords or phrases found",
                },
            },
            "required": [
                "threat_level",
                "threat_score",
                "is_directed",
                "confidence_in_direction",
                "reasoning",
                "keywords_detected",
            ],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_PROMPT = """\
You are a threat detection module in a personal safety wearable.
You receive a transcript with audio indicators and ASR confidence.

Assess whether the speech threatens the wearer's safety.

Key signals:
- Direct violence threats ("I will hurt you", "give me X or else")
- Directed vs media/TV/absent party — set is_directed accordingly
- Escalation: raised voice + threatening content = higher score
- Context: TV, quoting, recounting past events = NOT threats

False-positive patterns (score LOW/NONE):
- Figurative: "I could kill for a coffee" — no real danger
- Sarcasm: "Oh sure, I'll stab myself with this pen" — frustration, not threat
- Recounting: "He said he would hurt me" — report, not live threat
- Self-directed: "I want to die" — distress, not threat to others; is_directed=false

All transcripts are in English. If the text appears garbled, it is likely
a transcription error from accented speech — assess based on recognisable words.

ASR confidence note: low confidence means the transcript may be inaccurate.
Weight your assessment proportionally to ASR reliability.

Set confidence_in_direction to reflect how certain you are about who the \
speech is directed at (0.0 = no idea, 1.0 = certain).

Be conservative: false negatives are dangerous, but excessive false positives erode trust.
"""


class ThreatDetectorEngine(Engine[ThreatDetectorInput, ThreatAssessment]):
    """Assess threat level of a speech turn using GPT-4o-mini."""

    name = "threat_detector"

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self._client = client or _build_client(self.settings)

    def process(self, input_data: ThreatDetectorInput) -> ThreatAssessment:
        parts: list[str] = []

        if input_data.prior_context:
            parts.append("Previous turns:")
            for i, (text, score, level) in enumerate(input_data.prior_context, 1):
                parts.append(f"  [{i}] \"{text}\" (score={score:.2f}, level={level})")
            parts.append("")

        parts.append(f"Transcript: \"{input_data.transcription.cleaned_text}\"")
        parts.append(f"ASR confidence: {input_data.asr_confidence:.2f}")
        parts.append("")
        parts.append("Audio indicators:")
        parts.append(f"- is_loud: {input_data.features.is_loud}")
        parts.append(f"- is_sharp: {input_data.features.is_sharp}")
        parts.append(f"- RMS dB: {input_data.features.rms_db:.1f}")
        parts.append(f"- Spectral centroid: {input_data.features.spectral_centroid_mean:.0f} Hz")
        parts.append(f"- Spectral bandwidth: {input_data.features.spectral_bandwidth_mean:.0f} Hz")

        # Pass MFCC summary if available
        if input_data.features.mfcc_means:
            mfcc_summary = ", ".join(f"{v:.1f}" for v in input_data.features.mfcc_means[:5])
            parts.append(f"- MFCC (first 5): [{mfcc_summary}]")

        parts.append(f"- RMS normalized: {input_data.features.rms_normalized:.2f}")
        parts.append(f"- Centroid normalized: {input_data.features.centroid_normalized:.2f}")

        user_msg = "\n".join(parts) + "\n"

        response = self._client.chat.completions.create(
            model=self.settings.threat_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=_THREAT_SCHEMA,
            temperature=self.settings.threat_detector_temperature,
        )

        raw = json.loads(response.choices[0].message.content)

        return ThreatAssessment(
            turn_id=input_data.turn.turn_id,
            threat_level=ThreatLevel(raw["threat_level"]),
            threat_score=max(0.0, min(1.0, raw["threat_score"])),
            is_directed=raw["is_directed"],
            reasoning=raw["reasoning"],
            keywords_detected=raw["keywords_detected"],
            confidence_in_direction=max(0.0, min(1.0, raw["confidence_in_direction"])),
        )
