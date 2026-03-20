"""C#2 — Speculative parallel LLM call for borderline cases.

Same schema as ThreatDetectorEngine but with a less conservative prompt
and higher temperature. Launched speculatively in parallel; result is only
consumed when the primary detector returns UNCERTAIN.
"""

from __future__ import annotations

import json

from openai import OpenAI

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.engines.threat_detector import ThreatDetectorInput, _THREAT_SCHEMA, _build_client
from threat_pipeline.models import ThreatAssessment, ThreatLevel


_SPECULATIVE_SYSTEM_PROMPT = """\
You are a secondary threat detection module in a personal safety wearable.
The primary detector was uncertain about this transcript. Your job is to
provide a second opinion, erring slightly more toward caution.

Consider the same factors as primary detection but pay extra attention to:
- Subtle escalation patterns that might be missed
- Indirect threats or implied violence
- Contextual signals from prior turns
- Audio indicators suggesting heightened emotion

All transcripts are in English. If the text appears garbled, it is likely
a transcription error from accented speech — assess based on recognisable words.

ASR confidence note: low confidence means the transcript may be inaccurate.
Assess based on available evidence.

Set confidence_in_direction to reflect certainty about who speech targets.
"""


class SpeculativeDetectorEngine(Engine[ThreatDetectorInput, ThreatAssessment]):
    """Secondary LLM detector for borderline/uncertain cases."""

    name = "speculative_detector"

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
        parts.append(f"- RMS normalized: {input_data.features.rms_normalized:.2f}")

        user_msg = "\n".join(parts) + "\n"

        response = self._client.chat.completions.create(
            model=self.settings.threat_model,
            messages=[
                {"role": "system", "content": _SPECULATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=_THREAT_SCHEMA,
            temperature=self.settings.speculative_llm_temperature,
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
