"""CLI entry point for the eNOugh threat detection pipeline.

Usage:
    python -m threat_pipeline.cli run-all --audio-dir audio/
    python -m threat_pipeline.cli run --file audio/keyword_only.wav
    python -m threat_pipeline.cli benchmark --audio-dir audio/ --runs 10
"""

from __future__ import annotations

import threat_pipeline._suppress  # noqa: F401  — must run before torch/tf

import argparse
import json
import sys
from pathlib import Path

import torch

from threat_pipeline.config import get_settings
from threat_pipeline.models import AlertAction
from threat_pipeline.pipeline import ThreatPipeline


def _print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def _print_result(result) -> None:
    """Pretty-print a PipelineResult."""
    print(f"\n  Source: {result.source_path}")
    print(f"  Turns detected: {result.total_turns}")
    print(f"  Alerts fired: {result.alerts_fired}")
    print(f"  Total latency: {result.total_latency_s:.3f}s")

    if result.engine_timings:
        print("\n  Latency breakdown:")
        for engine, t in sorted(result.engine_timings.items()):
            print(f"    {engine:.<30s} {t:.3f}s")

    for tr in result.turn_results:
        _print_turn(tr)


def _print_turn(tr) -> None:
    """Pretty-print a single TurnResult."""
    turn = tr.turn
    print(f"\n  --- Turn {turn.turn_id} [{turn.start_s:.2f}s - {turn.end_s:.2f}s] ---")

    if tr.transcription:
        t = tr.transcription
        conf_marker = " [LOW ASR]" if t.low_asr_confidence else ""
        print(f"  Transcript: \"{t.cleaned_text}\"{conf_marker}")
        print(f"  ASR confidence: {t.asr_confidence:.2f}  no_speech_prob: {t.no_speech_prob:.2f}")

    if tr.features:
        f = tr.features
        print(f"  Audio: RMS={f.rms_db:.1f}dB  centroid={f.spectral_centroid_mean:.0f}Hz"
              f"  loud={f.is_loud}  sharp={f.is_sharp}")
        print(f"  Audio (continuous): rms_norm={f.rms_normalized:.2f}"
              f"  centroid_norm={f.centroid_normalized:.2f}"
              f"  bandwidth={f.spectral_bandwidth_mean:.0f}Hz")

    if tr.threat:
        t = tr.threat
        print(f"  Threat: {t.threat_level.value} (score={t.threat_score:.2f},"
              f" directed={t.is_directed},"
              f" confidence_in_direction={t.confidence_in_direction:.2f})")
        print(f"  Reasoning: {t.reasoning}")
        if t.keywords_detected:
            print(f"  Keywords: {', '.join(t.keywords_detected)}")

    if tr.text_classification:
        tc = tr.text_classification
        print(f"  Text classifier: {tc.label} (toxicity={tc.toxicity_score:.2f},"
              f" confidence={tc.confidence:.2f})")

    if tr.wav2vec2_result:
        w = tr.wav2vec2_result
        print(f"  Wav2Vec2 transcript: \"{w.ctc_transcript}\"")

    if tr.sound_events and tr.sound_events.events:
        events_str = ", ".join(f"{e['class']}={e['confidence']:.2f}" for e in tr.sound_events.events)
        print(f"  YAMNet events: {events_str}")

    if tr.incident_snapshot:
        snap = tr.incident_snapshot
        esc = " ** ESCALATING **" if snap.is_escalating else ""
        print(f"  Incident: accumulated={snap.accumulated_score:.2f}"
              f"  rising={snap.consecutive_rising}  turns={snap.turn_count}{esc}")

    if tr.decision:
        d = tr.decision
        action_display = d.action.value.upper()
        if d.suppressed:
            action_display += " (suppressed)"
        print(f"  Decision: {action_display} (final_score={d.final_score:.2f},"
              f" incident={d.incident_id})")

    if tr.alert:
        print(f"  ** ALERT PUBLISHED ** incident={tr.alert.incident_id}")


def _serialize_result(result) -> dict:
    """Convert a PipelineResult to a JSON-safe dict, excluding Tensor fields."""
    data = result.model_dump()
    _strip_tensors(data)
    return data


def _strip_tensors(obj):
    """Recursively remove torch.Tensor values and large arrays from nested dicts/lists."""
    if isinstance(obj, dict):
        keys_to_drop = [
            k for k, v in obj.items()
            if isinstance(v, torch.Tensor)
            or (k == "embedding" and isinstance(v, list))  # wav2vec2 768-dim vector
        ]
        for k in keys_to_drop:
            del obj[k]
        for v in obj.values():
            _strip_tensors(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_tensors(item)


def cmd_run(args: argparse.Namespace) -> None:
    """Process a single audio file."""
    settings = get_settings()
    pipeline = ThreatPipeline(settings)

    result = pipeline.process_file(args.file)

    if getattr(args, "json", False):
        print(json.dumps(_serialize_result(result), indent=2))
        return

    _print_header(f"Processing: {args.file}")
    _print_result(result)


def cmd_run_all(args: argparse.Namespace) -> None:
    """Process all WAV files in a directory."""
    audio_dir = Path(args.audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        sys.exit(1)

    settings = get_settings()
    pipeline = ThreatPipeline(settings)

    print(f"\nFound {len(wav_files)} WAV files in {audio_dir}")

    results = []
    for wav in wav_files:
        result = pipeline.process_file(str(wav))
        results.append(result)

        if not getattr(args, "json", False):
            _print_header(f"Processing: {wav.name}")
            _print_result(result)

    if getattr(args, "json", False):
        print(json.dumps([_serialize_result(r) for r in results], indent=2))
        return

    # Summary table
    _print_header("SUMMARY")
    print(f"\n  {'File':<30s} {'Turns':>5s} {'Alerts':>6s} {'Action':<12s} {'Score':>6s} {'Latency':>8s}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*12} {'-'*6} {'-'*8}")

    for r in results:
        name = Path(r.source_path).name
        if r.turn_results:
            actions = [tr.decision.action for tr in r.turn_results if tr.decision]
            top_action = max(actions, key=lambda a: list(AlertAction).index(a)) if actions else AlertAction.NO_ACTION
            scores = [tr.decision.final_score for tr in r.turn_results if tr.decision]
            max_score = max(scores) if scores else 0.0
        else:
            top_action = AlertAction.NO_ACTION
            max_score = 0.0

        print(f"  {name:<30s} {r.total_turns:>5d} {r.alerts_fired:>6d}"
              f" {top_action.value:<12s} {max_score:>6.2f} {r.total_latency_s:>7.3f}s")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark harness."""
    from threat_pipeline.benchmark import run_benchmark

    settings = get_settings()
    run_benchmark(
        audio_dir=args.audio_dir,
        num_runs=args.runs,
        output_dir=args.output_dir,
        settings=settings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="threat-pipeline",
        description="eNOugh — Real-time audio threat detection pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Process a single audio file")
    p_run.add_argument("--file", "-f", required=True, help="Path to WAV file")
    p_run.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    p_run.set_defaults(func=cmd_run)

    # run-all
    p_all = sub.add_parser("run-all", help="Process all WAVs in a directory")
    p_all.add_argument("--audio-dir", "-d", required=True, help="Directory containing WAV files")
    p_all.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    p_all.set_defaults(func=cmd_run_all)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run benchmark harness (10 runs × avg/stddev charts)")
    p_bench.add_argument("--audio-dir", "-d", required=True, help="Directory containing WAV files")
    p_bench.add_argument("--runs", "-r", type=int, default=10, help="Number of runs per file (default: 10)")
    p_bench.add_argument("--output-dir", "-o", default="benchmark_results", help="Output directory for charts")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
