"""Engine 2 — Voice Activity Detection using Silero VAD.

Segments audio into speech turns, discarding silence.  Silero VAD runs
in <1 ms per audio chunk, is PyTorch-native, and is ONNX-exportable —
ideal for edge deployment on the eNOugh wearable.

Production note: in streaming mode this would process fixed-size chunks
(e.g. 512 samples at 16 kHz = 32 ms) and emit turn boundaries via a
state machine, rather than processing the full file at once.
"""

from __future__ import annotations

import torch

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import AudioSegment, SpeechTurn


class VADEngine(Engine[AudioSegment, list[SpeechTurn]]):
    """Segment audio into speech turns using Silero VAD."""

    name = "vad"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self._model = None
        self._get_speech_timestamps = None

    def _load_model(self) -> None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]  # get_speech_timestamps

    def process(self, segment: AudioSegment) -> list[SpeechTurn]:
        if self._model is None:
            self._load_model()

        samples: torch.Tensor = segment.samples
        sr = segment.sample_rate

        timestamps = self._get_speech_timestamps(
            samples,
            self._model,
            sampling_rate=sr,
            threshold=self.settings.vad_threshold,
            min_speech_duration_ms=self.settings.vad_min_speech_duration_ms,
            min_silence_duration_ms=self.settings.vad_min_silence_duration_ms,
        )

        turns: list[SpeechTurn] = []
        for i, ts in enumerate(timestamps):
            start_sample = ts["start"]
            end_sample = ts["end"]
            turn_audio = samples[start_sample:end_sample]

            turns.append(
                SpeechTurn(
                    turn_id=i,
                    start_s=start_sample / sr,
                    end_s=end_sample / sr,
                    audio_samples=turn_audio,
                    sample_rate=sr,
                )
            )

        return turns
