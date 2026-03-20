"""C#2 — Benchmark harness for pipeline evaluation.

Runs the pipeline on all WAV files multiple times, collects per-run metrics,
computes mean/stddev, and generates matplotlib charts comparing GPT-4o-mini
threat_score vs DistilRoBERTa toxicity_score.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from threat_pipeline.config import Settings, get_settings
from threat_pipeline.models import PipelineResult
from threat_pipeline.pipeline import ThreatPipeline


@dataclass
class RunMetrics:
    """Metrics collected from a single pipeline run on one file."""

    file_name: str
    final_score: float
    action: str
    total_latency_s: float
    engine_latencies: dict[str, float]
    threat_score: float
    text_classifier_toxicity: float
    yamnet_max_confidence: float
    wav2vec2_transcript: str
    asr_confidence: float


@dataclass
class FileMetrics:
    """Aggregated metrics for one file across multiple runs."""

    file_name: str
    runs: list[RunMetrics] = field(default_factory=list)

    @property
    def final_scores(self) -> list[float]:
        return [r.final_score for r in self.runs]

    @property
    def latencies(self) -> list[float]:
        return [r.total_latency_s for r in self.runs]

    @property
    def threat_scores(self) -> list[float]:
        return [r.threat_score for r in self.runs]

    @property
    def toxicity_scores(self) -> list[float]:
        return [r.text_classifier_toxicity for r in self.runs]


def _extract_run_metrics(file_name: str, result: PipelineResult) -> RunMetrics:
    """Extract the most interesting metrics from a PipelineResult."""
    # Use the highest-scoring turn as representative
    max_final = 0.0
    max_action = "no_action"
    max_threat = 0.0
    max_toxicity = 0.0
    max_yamnet = 0.0
    wav2vec2_text = ""
    asr_conf = 1.0

    for tr in result.turn_results:
        if tr.decision and tr.decision.final_score > max_final:
            max_final = tr.decision.final_score
            max_action = tr.decision.action.value
        if tr.threat and tr.threat.threat_score > max_threat:
            max_threat = tr.threat.threat_score
        if tr.text_classification and tr.text_classification.toxicity_score > max_toxicity:
            max_toxicity = tr.text_classification.toxicity_score
        if tr.sound_events and tr.sound_events.max_threat_event_confidence > max_yamnet:
            max_yamnet = tr.sound_events.max_threat_event_confidence
        if tr.wav2vec2_result and not wav2vec2_text:
            wav2vec2_text = tr.wav2vec2_result.ctc_transcript
        if tr.transcription:
            asr_conf = min(asr_conf, tr.transcription.asr_confidence)

    return RunMetrics(
        file_name=file_name,
        final_score=max_final,
        action=max_action,
        total_latency_s=result.total_latency_s,
        engine_latencies=dict(result.engine_timings),
        threat_score=max_threat,
        text_classifier_toxicity=max_toxicity,
        yamnet_max_confidence=max_yamnet,
        wav2vec2_transcript=wav2vec2_text,
        asr_confidence=asr_conf,
    )


def run_benchmark(
    audio_dir: str,
    num_runs: int = 10,
    output_dir: str = "benchmark_results",
    settings: Settings | None = None,
) -> dict[str, FileMetrics]:
    """Run the benchmark: process each WAV file num_runs times."""
    audio_path = Path(audio_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audio_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {audio_dir}")

    settings = settings or get_settings()
    all_metrics: dict[str, FileMetrics] = {}

    for wav in wav_files:
        file_name = wav.name
        fm = FileMetrics(file_name=file_name)
        print(f"\nBenchmarking {file_name} ({num_runs} runs)...")

        for run_idx in range(num_runs):
            # Fresh pipeline per run for clean state
            pipeline = ThreatPipeline(settings)
            result = pipeline.process_file(str(wav))
            metrics = _extract_run_metrics(file_name, result)
            fm.runs.append(metrics)
            print(f"  Run {run_idx + 1}/{num_runs}: score={metrics.final_score:.3f} "
                  f"action={metrics.action} latency={metrics.total_latency_s:.3f}s")

        all_metrics[file_name] = fm

    # Generate charts
    _generate_charts(all_metrics, out_path)

    # Save raw metrics as JSON
    _save_raw_metrics(all_metrics, out_path)

    return all_metrics


def _generate_charts(metrics: dict[str, FileMetrics], out_dir: Path) -> None:
    """Generate all benchmark charts."""
    file_names = list(metrics.keys())

    # 1. Bar chart: avg final_score per file with stddev error bars
    fig, ax = plt.subplots(figsize=(10, 5))
    means = [np.mean(metrics[f].final_scores) for f in file_names]
    stds = [np.std(metrics[f].final_scores) for f in file_names]
    x = np.arange(len(file_names))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".wav", "") for f in file_names], rotation=30, ha="right")
    ax.set_ylabel("Final Score")
    ax.set_title("Average Final Score per File (with stddev)")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "final_scores.png", dpi=150)
    plt.close(fig)

    # 2. Bar chart: avg latency per engine
    all_engine_names: set[str] = set()
    for fm in metrics.values():
        for run in fm.runs:
            all_engine_names.update(run.engine_latencies.keys())

    engine_names = sorted(all_engine_names)
    engine_means = []
    for eng in engine_names:
        vals = []
        for fm in metrics.values():
            for run in fm.runs:
                if eng in run.engine_latencies:
                    vals.append(run.engine_latencies[eng])
        engine_means.append(np.mean(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(engine_names))
    ax.bar(x, engine_means, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(engine_names, rotation=30, ha="right")
    ax.set_ylabel("Avg Latency (s)")
    ax.set_title("Average Latency per Engine (across all files/runs)")
    fig.tight_layout()
    fig.savefig(out_dir / "engine_latencies.png", dpi=150)
    plt.close(fig)

    # 3. Comparison: GPT-4o-mini threat_score vs DistilRoBERTa toxicity_score
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    threat_means = [np.mean(metrics[f].threat_scores) for f in file_names]
    toxic_means = [np.mean(metrics[f].toxicity_scores) for f in file_names]
    threat_stds = [np.std(metrics[f].threat_scores) for f in file_names]
    toxic_stds = [np.std(metrics[f].toxicity_scores) for f in file_names]
    x = np.arange(len(file_names))
    ax.bar(x - width / 2, threat_means, width, yerr=threat_stds, label="GPT-4o-mini threat_score",
           capsize=3, color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, toxic_means, width, yerr=toxic_stds, label="DistilRoBERTa toxicity",
           capsize=3, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".wav", "") for f in file_names], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("GPT-4o-mini vs DistilRoBERTa: Score Comparison (C#7)")
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "threat_vs_toxicity.png", dpi=150)
    plt.close(fig)

    # 4. Latency distribution histogram
    all_latencies = []
    for fm in metrics.values():
        all_latencies.extend(fm.latencies)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_latencies, bins=20, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Total Latency (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Latency Distribution (all files, all runs)")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nCharts saved to {out_dir}/")


def _save_raw_metrics(metrics: dict[str, FileMetrics], out_dir: Path) -> None:
    """Save raw benchmark data as JSON."""
    data = {}
    for fname, fm in metrics.items():
        data[fname] = {
            "runs": [
                {
                    "final_score": r.final_score,
                    "action": r.action,
                    "total_latency_s": r.total_latency_s,
                    "threat_score": r.threat_score,
                    "text_classifier_toxicity": r.text_classifier_toxicity,
                    "yamnet_max_confidence": r.yamnet_max_confidence,
                    "wav2vec2_transcript": r.wav2vec2_transcript,
                    "asr_confidence": r.asr_confidence,
                }
                for r in fm.runs
            ],
            "summary": {
                "final_score_mean": float(np.mean(fm.final_scores)),
                "final_score_std": float(np.std(fm.final_scores)),
                "latency_mean": float(np.mean(fm.latencies)),
                "latency_std": float(np.std(fm.latencies)),
                "threat_score_mean": float(np.mean(fm.threat_scores)),
                "toxicity_score_mean": float(np.mean(fm.toxicity_scores)),
            },
        }

    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Raw metrics saved to {out_dir / 'benchmark_results.json'}")
