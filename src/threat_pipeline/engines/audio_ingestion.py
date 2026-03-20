"""Engine 1 — Load WAV and resample to canonical 16 kHz mono.

Uses torchaudio (PyTorch-native) for resampling.  File I/O uses the
soundfile backend for cross-platform reliability.

Production note: in a streaming deployment this would accept chunked PCM
from the wearable's BLE/WebSocket feed instead of file paths.
"""

from __future__ import annotations

import soundfile as sf
import numpy as np
import torch
import torchaudio

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import AudioSegment


class AudioIngestionEngine(Engine[str, AudioSegment]):
    """Load a WAV file and resample to target sample rate."""

    name = "audio_ingestion"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()

    def process(self, file_path: str) -> AudioSegment:
        # Use soundfile for reliable cross-platform WAV loading
        data, sr = sf.read(file_path, dtype="float32")

        # Convert to torch tensor
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
        else:
            # Multi-channel: (N, C) → (C, N), then average to mono
            waveform = torch.from_numpy(data.T)
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target rate (16 kHz for Silero VAD)
        target_sr = self.settings.target_sample_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
            waveform = resampler(waveform)

        # Squeeze to 1-D tensor
        waveform = waveform.squeeze(0)
        duration_s = waveform.shape[0] / target_sr

        return AudioSegment(
            samples=waveform,
            sample_rate=target_sr,
            duration_s=duration_s,
            source_path=file_path,
        )
