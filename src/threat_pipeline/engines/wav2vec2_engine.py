"""C#8 — Wav2Vec2 secondary ASR + embedding extraction.

Uses facebook/wav2vec2-base-960h for dual output:
  1. CTC-decoded transcript (secondary ASR, comparison/fallback)
  2. Hidden-state embedding (768-dim, for future learned fusion)

Follows the lazy-loading pattern from vad.py.
"""

from __future__ import annotations

import numpy as np
import torch

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import SpeechTurn, Wav2Vec2Result


class Wav2Vec2Engine(Engine[SpeechTurn, Wav2Vec2Result]):
    """Extract CTC transcript and embedding from wav2vec2-base-960h."""

    name = "wav2vec2"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self._model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h", use_safetensors=False,
        )
        self._model.eval()

    def process(self, turn: SpeechTurn) -> Wav2Vec2Result:
        if self._model is None:
            self._load_model()

        samples: torch.Tensor = turn.audio_samples
        audio_np = samples.numpy().astype(np.float32)

        # Processor expects 16kHz audio
        inputs = self._processor(
            audio_np,
            sampling_rate=turn.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(
                inputs.input_values,
                output_hidden_states=True,
            )

        # CTC transcript: argmax(logits) → decode
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        ctc_transcript = self._processor.batch_decode(predicted_ids)[0]

        # Embedding: mean-pool last hidden state → 768-dim
        last_hidden = outputs.hidden_states[-1]  # (1, T, 768)
        embedding = last_hidden.mean(dim=1).squeeze(0)  # (768,)
        embedding_list = embedding.tolist()

        return Wav2Vec2Result(
            turn_id=turn.turn_id,
            ctc_transcript=ctc_transcript,
            embedding=embedding_list,
        )
