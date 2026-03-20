"""Engine 4 — Deterministic audio-level feature extraction.

Extracts RMS (loudness), spectral centroid (sharpness/brightness),
zero-crossing rate, MFCCs, spectral contrast, mel spectrogram stats,
and spectral bandwidth from a speech turn using librosa.

C#5 upgrade: expanded feature set with continuous normalized values
for improved fusion signal quality.

Production note: these features are cheap (~1-3 ms per turn) and could run
on-device via NumPy/C++ without an ML runtime.
"""

from __future__ import annotations

import numpy as np
import librosa
import torch

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.models import AudioFeatures, SpeechTurn


def _sigmoid_normalize(value: float, center: float, scale: float) -> float:
    """Map a raw value to [0, 1] via sigmoid centered at `center`."""
    return 1.0 / (1.0 + np.exp(-(value - center) / scale))


class AudioFeaturesEngine(Engine[SpeechTurn, AudioFeatures]):
    """Extract audio features from a speech turn."""

    name = "audio_features"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()

    def process(self, turn: SpeechTurn) -> AudioFeatures:
        samples: torch.Tensor = turn.audio_samples
        y = samples.numpy().astype(np.float32)
        sr = turn.sample_rate
        s = self.settings

        # RMS → dB
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_db = float(librosa.amplitude_to_db(np.array([rms_mean]))[0])

        # Spectral centroid (brightness / sharpness proxy)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = float(np.mean(centroid))

        # Zero-crossing rate (rough proxy for noisiness / fricatives)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_mean = float(np.mean(zcr))

        # Boolean flags (backward compat)
        is_loud = rms_db >= s.loud_rms_db
        is_sharp = centroid_mean >= s.sharp_centroid_hz

        # ── C#5: Expanded features ──────────────────────────────────────

        # MFCCs (n_mfcc coefficients, mean across time)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=s.n_mfcc)
        mfcc_means = [float(np.mean(mfcc[i])) for i in range(s.n_mfcc)]

        # Spectral contrast (7 bands by default)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = [float(np.mean(contrast[i])) for i in range(contrast.shape[0])]

        # Mel spectrogram stats
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_spectrogram_mean = float(np.mean(mel_db))
        mel_spectrogram_std = float(np.std(mel_db))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_bandwidth_mean = float(np.mean(bandwidth))

        # Continuous normalized signals for fusion (C#5)
        # rms_normalized: sigmoid-scaled from rms_db, center=-20dB, scale=5
        rms_normalized = float(_sigmoid_normalize(rms_db, center=-20.0, scale=5.0))
        # centroid_normalized: sigmoid-scaled from centroid_mean, center=3000Hz, scale=500
        centroid_normalized = float(_sigmoid_normalize(centroid_mean, center=3000.0, scale=500.0))

        return AudioFeatures(
            turn_id=turn.turn_id,
            rms_db=rms_db,
            spectral_centroid_mean=centroid_mean,
            zero_crossing_rate=zcr_mean,
            is_loud=is_loud,
            is_sharp=is_sharp,
            mfcc_means=mfcc_means,
            spectral_contrast_mean=spectral_contrast_mean,
            mel_spectrogram_mean=mel_spectrogram_mean,
            mel_spectrogram_std=mel_spectrogram_std,
            spectral_bandwidth_mean=spectral_bandwidth_mean,
            rms_normalized=rms_normalized,
            centroid_normalized=centroid_normalized,
        )
