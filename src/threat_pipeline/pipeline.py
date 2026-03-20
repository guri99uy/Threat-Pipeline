"""Pipeline orchestrator — chains all engines for end-to-end processing.

Processes a WAV file through: ingest → VAD → concurrent feature/ASR/wav2vec2/yamnet
→ concurrent threat/text_classifier/speculative → alert decision → alert publish.

C#2/C#6 upgrade: three-input fusion with parallel execution groups,
IncidentState tracking, and speculative LLM for UNCERTAIN resolution.

Latency optimisations:
  - Eager model preloading (wav2vec2, text_classifier, yamnet) in parallel
  - Persistent ThreadPoolExecutor (one pool, reused across turns)
  - Conditional speculative LLM (only when prior context suggests borderline)
  - Wall-clock timing (not sum of parallel engine times)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.engine_base import Engine
from threat_pipeline.engines.alert_decision import AlertDecisionEngine, AlertDecisionInput
from threat_pipeline.engines.alert_publisher import AlertPublisherEngine, AlertPublisherInput
from threat_pipeline.engines.audio_features import AudioFeaturesEngine
from threat_pipeline.engines.audio_ingestion import AudioIngestionEngine
from threat_pipeline.engines.speculative_detector import SpeculativeDetectorEngine
from threat_pipeline.engines.text_classifier_engine import TextClassifierEngine
from threat_pipeline.engines.threat_detector import ThreatDetectorEngine, ThreatDetectorInput
from threat_pipeline.engines.transcription import TranscriptionEngine
from threat_pipeline.engines.vad import VADEngine
from threat_pipeline.engines.wav2vec2_engine import Wav2Vec2Engine
from threat_pipeline.engines.yamnet_engine import YAMNetEngine
from threat_pipeline.event_bus import EventBus
from threat_pipeline.incident_state import IncidentState
from threat_pipeline.models import (
    AlertAction,
    PipelineResult,
    TurnResult,
)

logger = logging.getLogger(__name__)

# Score thresholds for conditional speculative LLM firing.
# Only fire when the prior context suggests a borderline situation.
_SPEC_PRIOR_SCORE_THRESHOLD = 0.15


class ThreatPipeline:
    """End-to-end threat detection pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.event_bus = EventBus()

        # Core engines
        self.ingest = AudioIngestionEngine(self.settings)
        self.vad = VADEngine(self.settings)
        self.transcribe = TranscriptionEngine(self.settings)
        self.features = AudioFeaturesEngine(self.settings)
        self.threat_detector = ThreatDetectorEngine(self.settings)
        self.alert_decision = AlertDecisionEngine(self.settings)
        self.alert_publisher = AlertPublisherEngine(self.event_bus)

        # New engines (C#7, C#8, C#9)
        self.wav2vec2 = Wav2Vec2Engine(self.settings)
        self.text_classifier = TextClassifierEngine(self.settings)
        self.yamnet = YAMNetEngine(self.settings)
        self.speculative_detector = SpeculativeDetectorEngine(self.settings)

        # C#1: Cross-turn incident tracking
        self.incident_state = IncidentState(self.settings)

        # Persistent thread pool (avoids create/destroy per group per turn)
        self._pool = ThreadPoolExecutor(max_workers=5)

        # Eager-load local ML models in parallel (shifts ~28s from first turn to init)
        self._preload_models()

    def _preload_models(self) -> None:
        """Load wav2vec2, text_classifier, and yamnet models in parallel."""
        futures = [
            self._pool.submit(self._safe_load, self.wav2vec2, "wav2vec2"),
            self._pool.submit(self._safe_load, self.text_classifier, "text_classifier"),
            self._pool.submit(self._safe_load, self.yamnet, "yamnet"),
        ]
        for f in futures:
            f.result()  # wait for all to finish

    @staticmethod
    def _safe_load(engine, name: str) -> None:
        """Call an engine's _load_model if it has one."""
        try:
            if hasattr(engine, "_load_model") and callable(engine._load_model):
                engine._load_model()
        except Exception as e:
            logger.warning("Preload %s failed: %s", name, e)

    def _should_fire_speculative(self, prior_context: list[tuple[str, float, str]]) -> bool:
        """Decide whether to fire the speculative LLM on this turn.

        Only fire when:
          - speculative_llm_enabled is True, AND
          - there's prior context suggesting a non-trivial situation (any prior
            score above the threshold), OR this is the first turn (no context yet,
            we don't know what to expect).
        """
        if not self.settings.speculative_llm_enabled:
            return False
        if not prior_context:
            # First turn — be cautious, fire speculative
            return True
        return any(score >= _SPEC_PRIOR_SCORE_THRESHOLD for _, score, _ in prior_context)

    def process_file(self, file_path: str) -> PipelineResult:
        """Run the full pipeline on a single audio file."""
        wall_start = time.perf_counter()
        timings: dict[str, float] = {}

        # 1. Ingest
        segment = self.ingest.run(file_path)
        timings["audio_ingestion"] = self.ingest.last_latency_s

        # 2. VAD
        turns = self.vad.run(segment)
        timings["vad"] = self.vad.last_latency_s

        result = PipelineResult(
            source_path=file_path,
            total_turns=len(turns),
        )

        if not turns:
            result.engine_timings = timings
            result.total_latency_s = time.perf_counter() - wall_start
            return result

        # Sliding window of prior context: (text, score, threat_level)
        prior_context: list[tuple[str, float, str]] = []

        # 3-7. Process each turn
        for turn in turns:
            turn_timings: dict[str, float] = {}

            # ── Group 1: parallel (all take SpeechTurn) ─────────────────
            feat_future = self._pool.submit(self.features.run, turn)
            trans_future = self._pool.submit(self.transcribe.run, turn)
            wav2vec2_future = self._pool.submit(self._safe_run, self.wav2vec2, turn, "wav2vec2")
            yamnet_future = self._pool.submit(self._safe_run, self.yamnet, turn, "yamnet")

            feat = feat_future.result()
            transcript = trans_future.result()
            wav2vec2_result = wav2vec2_future.result()
            sound_events = yamnet_future.result()

            turn_timings["audio_features"] = self.features.last_latency_s
            turn_timings["transcription"] = self.transcribe.last_latency_s
            turn_timings["wav2vec2"] = self.wav2vec2.last_latency_s
            turn_timings["yamnet"] = self.yamnet.last_latency_s

            # ── Group 2: need transcript (parallel) ─────────────────────
            threat_input = ThreatDetectorInput(
                turn, transcript, feat,
                prior_context=list(prior_context),
                asr_confidence=transcript.asr_confidence,
            )

            threat_future = self._pool.submit(self.threat_detector.run, threat_input)
            text_cls_future = self._pool.submit(
                self._safe_run, self.text_classifier, transcript, "text_classifier"
            )

            # Conditional speculative LLM — skip when clearly benign
            spec_future: Future | None = None
            if self._should_fire_speculative(prior_context):
                spec_future = self._pool.submit(
                    self._safe_run, self.speculative_detector, threat_input, "speculative_detector"
                )

            threat = threat_future.result()
            text_classification = text_cls_future.result()
            speculative_result = spec_future.result() if spec_future else None

            turn_timings["threat_detector"] = self.threat_detector.last_latency_s
            turn_timings["text_classifier"] = self.text_classifier.last_latency_s
            if spec_future:
                turn_timings["speculative_detector"] = self.speculative_detector.last_latency_s

            # Update prior context window
            prior_context.append((
                transcript.cleaned_text,
                threat.threat_score,
                threat.threat_level.value,
            ))
            if len(prior_context) > self.settings.context_window_size:
                prior_context.pop(0)

            # ── C#1: IncidentState update ───────────────────────────────
            incident_snapshot = self.incident_state.update(file_path, threat.threat_score)

            # ── Group 3: sequential (alert decision + publish) ──────────
            decision_input = AlertDecisionInput(
                threat=threat,
                features=feat,
                source_id=file_path,
                text_classification=text_classification,
                wav2vec2_result=wav2vec2_result,
                sound_events=sound_events,
                asr_confidence=transcript.asr_confidence,
                incident_snapshot=incident_snapshot,
            )
            decision = self.alert_decision.run(decision_input)
            turn_timings["alert_decision"] = self.alert_decision.last_latency_s

            # ── C#2: If UNCERTAIN, consult speculative result ───────────
            if decision.action == AlertAction.UNCERTAIN and speculative_result is not None:
                spec_decision_input = AlertDecisionInput(
                    threat=speculative_result,
                    features=feat,
                    source_id=file_path,
                    text_classification=text_classification,
                    wav2vec2_result=wav2vec2_result,
                    sound_events=sound_events,
                    asr_confidence=transcript.asr_confidence,
                    incident_snapshot=incident_snapshot,
                )
                spec_decision = self.alert_decision.run(spec_decision_input)
                if _action_rank(spec_decision.action) > _action_rank(decision.action):
                    decision = spec_decision
                    threat = speculative_result

            # Alert publishing
            pub_input = AlertPublisherInput(
                decision=decision,
                transcription=transcript,
                threat=threat,
                features=feat,
                latency_breakdown={**timings, **turn_timings},
                text_classification=text_classification,
                wav2vec2_result=wav2vec2_result,
                sound_events=sound_events,
                incident_snapshot=incident_snapshot,
            )
            alert = self.alert_publisher.run(pub_input)
            turn_timings["alert_publisher"] = self.alert_publisher.last_latency_s

            turn_result = TurnResult(
                turn=turn,
                transcription=transcript,
                features=feat,
                threat=threat,
                decision=decision,
                alert=alert,
                wav2vec2_result=wav2vec2_result,
                text_classification=text_classification,
                sound_events=sound_events,
                incident_snapshot=incident_snapshot,
            )
            result.turn_results.append(turn_result)

            if decision.action in (AlertAction.ALERT, AlertAction.ESCALATE) and not decision.suppressed:
                result.alerts_fired += 1

            # Accumulate per-turn timings
            for k, v in turn_timings.items():
                timings[k] = timings.get(k, 0.0) + v

        result.engine_timings = timings
        result.total_latency_s = time.perf_counter() - wall_start
        return result

    @staticmethod
    def _safe_run(engine: Engine, input_data, engine_name: str):
        """Run an engine, returning None on failure (graceful degradation)."""
        try:
            return engine.run(input_data)
        except Exception as e:
            logger.warning("Engine %s failed: %s", engine_name, e)
            return None


def _action_rank(action: AlertAction) -> int:
    """Numeric rank for AlertAction comparison."""
    return list(AlertAction).index(action)
