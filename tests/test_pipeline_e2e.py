"""End-to-end pipeline tests."""

from pathlib import Path

import pytest

from threat_pipeline.models import AlertAction
from threat_pipeline.pipeline import ThreatPipeline

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
WAV_FILES = sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []


@pytest.mark.api
class TestPipelineE2E:
    """Full pipeline tests — require OpenAI API key."""

    @pytest.fixture
    def pipeline(self):
        from threat_pipeline.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return ThreatPipeline(settings)

    @pytest.mark.parametrize("wav", WAV_FILES, ids=[w.name for w in WAV_FILES])
    def test_processes_file(self, pipeline, wav):
        """Each WAV should process without errors."""
        result = pipeline.process_file(str(wav))
        assert result.total_turns >= 1
        assert result.total_latency_s > 0
        assert len(result.turn_results) == result.total_turns

    def test_keyword_only_triggers_alert(self, pipeline):
        """keyword_only.wav should produce ALERT or ESCALATE."""
        wav = AUDIO_DIR / "keyword_only.wav"
        if not wav.exists():
            pytest.skip("keyword_only.wav not found")

        result = pipeline.process_file(str(wav))
        actions = [tr.decision.action for tr in result.turn_results if tr.decision]
        top = max(actions, key=lambda a: list(AlertAction).index(a))
        assert top in (AlertAction.ALERT, AlertAction.ESCALATE)

    def test_casual_chat_no_action(self, pipeline):
        """casual_chat.wav should produce NO_ACTION."""
        wav = AUDIO_DIR / "casual_chat.wav"
        if not wav.exists():
            pytest.skip("casual_chat.wav not found")

        result = pipeline.process_file(str(wav))
        actions = [tr.decision.action for tr in result.turn_results if tr.decision]
        # All turns should be NO_ACTION, LOG, or UNCERTAIN at most
        for action in actions:
            assert action in (AlertAction.NO_ACTION, AlertAction.LOG, AlertAction.UNCERTAIN)

    def test_latency_breakdown(self, pipeline):
        """Latency breakdown should include all engine names."""
        wav = WAV_FILES[0] if WAV_FILES else None
        if wav is None:
            pytest.skip("No WAV files")

        result = pipeline.process_file(str(wav))
        expected_engines = {"audio_ingestion", "vad", "audio_features",
                           "transcription", "threat_detector", "alert_decision",
                           "alert_publisher", "wav2vec2", "yamnet",
                           "text_classifier"}
        assert expected_engines.issubset(set(result.engine_timings.keys()))

    def test_new_turn_result_fields(self, pipeline):
        """TurnResult should include new engine outputs."""
        wav = WAV_FILES[0] if WAV_FILES else None
        if wav is None:
            pytest.skip("No WAV files")

        result = pipeline.process_file(str(wav))
        for tr in result.turn_results:
            # ASR confidence should be set
            if tr.transcription:
                assert 0.0 <= tr.transcription.asr_confidence <= 1.0

            # Expanded audio features
            if tr.features:
                assert len(tr.features.mfcc_means) == 13
                assert 0.0 <= tr.features.rms_normalized <= 1.0

            # Threat confidence_in_direction
            if tr.threat:
                assert 0.0 <= tr.threat.confidence_in_direction <= 1.0

            # Incident snapshot should be present
            assert tr.incident_snapshot is not None
