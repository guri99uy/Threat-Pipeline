"""C#7 — Pre-trained DistilRoBERTa toxicity classifier.

Runs a HuggingFace text-classification pipeline (s-nlp/roberta_toxicity_classifier)
as a parallel branch to GPT-4o-mini threat detection. Fast local inference
(~20-50ms) provides a second opinion for the fusion layer.

Compare vs GPT-4o-mini in benchmarks to calibrate weight_text_classifier.

Follows the lazy-loading pattern from vad.py.
"""

from __future__ import annotations

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import TextClassification, TranscriptionResult


class TextClassifierEngine(Engine[TranscriptionResult, TextClassification]):
    """Classify transcript toxicity using DistilRoBERTa."""

    name = "text_classifier"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self._pipeline = None

    def _load_model(self) -> None:
        from transformers import pipeline as hf_pipeline

        self._pipeline = hf_pipeline(
            "text-classification",
            model="s-nlp/roberta_toxicity_classifier",
            truncation=True,
            model_kwargs={"use_safetensors": False},
        )

    def process(self, transcript: TranscriptionResult) -> TextClassification:
        if self._pipeline is None:
            self._load_model()

        text = transcript.cleaned_text
        if not text:
            return TextClassification(
                turn_id=transcript.turn_id,
                toxicity_score=0.0,
                label="neutral",
                confidence=1.0,
            )

        results = self._pipeline(text)
        top = results[0]

        label = top["label"]
        confidence = top["score"]

        # The model returns "toxic" or "neutral". Map to toxicity_score.
        if label.lower() == "toxic":
            toxicity_score = confidence
        else:
            toxicity_score = 1.0 - confidence

        return TextClassification(
            turn_id=transcript.turn_id,
            toxicity_score=toxicity_score,
            label=label,
            confidence=confidence,
        )
