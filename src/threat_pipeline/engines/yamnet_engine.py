"""C#9 — YAMNet sound event detection.

Uses Google's YAMNet (TensorFlow Hub) to detect threat-related sound events
like gunshots, explosions, glass breaking, and screaming.

Non-decisive fusion signal: boosts the final score but never overrides
other signals on its own.

Follows the lazy-loading pattern from vad.py.
"""

from __future__ import annotations

import numpy as np
import torch

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import SoundEventResult, SpeechTurn


class YAMNetEngine(Engine[SpeechTurn, SoundEventResult]):
    """Detect threat-related sound events using YAMNet."""

    name = "yamnet"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self._model = None
        self._class_names: list[str] = []

    def _load_model(self) -> None:
        import tensorflow_hub as hub
        import csv
        import os
        import tensorflow as tf

        self._model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load class map from the model's assets
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path) as f:
            reader = csv.DictReader(f)
            self._class_names = [row["display_name"] for row in reader]

    def process(self, turn: SpeechTurn) -> SoundEventResult:
        if self._model is None:
            self._load_model()

        samples: torch.Tensor = turn.audio_samples
        audio_np = samples.numpy().astype(np.float32)

        # YAMNet expects 16kHz mono float32
        # If sample rate differs, resample (should already be 16kHz from pipeline)
        if turn.sample_rate != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=turn.sample_rate, target_sr=16000)

        scores, embeddings, spectrogram = self._model(audio_np)
        scores_np = scores.numpy()

        # Find threat-related events
        s = self.settings
        threat_classes_lower = [c.lower() for c in s.yamnet_threat_classes]
        events: list[dict] = []
        max_threat_confidence = 0.0

        # Average scores across time frames
        mean_scores = np.mean(scores_np, axis=0) if len(scores_np.shape) > 1 else scores_np

        for idx, class_name in enumerate(self._class_names):
            if idx >= len(mean_scores):
                break
            score = float(mean_scores[idx])
            if score >= s.yamnet_confidence_threshold:
                # Check if this class matches any threat class
                for threat_class in s.yamnet_threat_classes:
                    if threat_class.lower() in class_name.lower() or class_name.lower() in threat_class.lower():
                        events.append({"class": class_name, "confidence": score})
                        max_threat_confidence = max(max_threat_confidence, score)
                        break

        return SoundEventResult(
            turn_id=turn.turn_id,
            events=events,
            max_threat_event_confidence=max_threat_confidence,
            has_threat_sound=len(events) > 0,
        )
