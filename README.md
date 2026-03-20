# Real-time Audio Threat Detection Pipeline

A modular threat detection pipeline that processes audio from a wearable device, identifies threatening speech, and publishes structured alerts for operators at an Alarm Receiving Centre (ARC). Built as a working prototype with all engines wired and producing output — ready for calibration with labelled data.

For design rationale, known weaknesses, and the calibration roadmap, see [ANALYSIS.md](ANALYSIS.md).

## Architecture

```
WAV File
  │
  ▼
┌─────────────────┐
│ 1. AudioIngest   │  soundfile + torchaudio → 16 kHz mono
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. VAD           │  Silero VAD → list[SpeechTurn]
└────────┬────────┘
         ▼  (per turn)
   ┌─────┼──────────┬──────────────┐
   ▼     ▼          ▼              ▼          Group 1 (parallel)
┌──────┐┌──────┐ ┌──────────┐ ┌──────────┐
│Trans-││Audio │ │ Wav2Vec2 │ │  YAMNet  │
│cribe ││Feats │ │          │ │          │
└──┬───┘└──┬───┘ └────┬─────┘ └────┬─────┘
   └───┬───┘          │            │
       ▼              ▼            ▼
   ┌───┼──────────────┘            │
   ▼   ▼                          │          Group 2 (parallel, needs transcript)
┌──────────┐ ┌────────────┐ ┌─────┴─────┐
│  Threat  │ │TextClassify│ │Speculative│
│  Detect  │ │(RoBERTa)   │ │   LLM     │
└────┬─────┘ └─────┬──────┘ └─────┬─────┘
     └──────┬──────┘              │
            ▼                     │          Group 3 (sequential)
┌───────────────────┐             │
│  AlertDecision     │◄───────────┘  (consumed only on UNCERTAIN)
│  + IncidentState   │
└────────┬──────────┘
         ▼
┌─────────────────┐
│  AlertPublish    │  EventBus → "alerts" | "review"
└─────────────────┘
```

The multi-engine approach is grounded in research: the Multimodal Audio Violence Detection paper demonstrates that learned fusion of independent branches (audio + text) significantly outperforms any single-path architecture because the relative reliability of each modality varies by input. We follow this principle — six engines produce independent signals, fused into a single decision.

## Design decisions

| Decision | Why |
|---|---|
| **Multi-engine parallel fusion** | No single model is sufficient. Audio features, text semantics, and sound events each catch different threat types. Research confirms learned fusion outperforms single-path (see ANALYSIS.md). |
| **Whisper API with verbose_json** | Extracts `avg_logprob` → ASR confidence and `no_speech_prob`. Confidence gates the fusion: `effective_threat = threat_score × asr_confidence`. Prevents hallucination-driven false alerts. |
| **GPT-4o-mini + Structured Outputs** | Fast, cheap, schema-enforced JSON. Lower temperature (0.05) for conservative assessment. Includes `confidence_in_direction` to express attribution uncertainty. |
| **DistilRoBERTa toxicity classifier** | ~20ms local inference, runs parallel to the LLM. Wired and producing output; weight set to 0.0 pending calibration with labelled data. |
| **wav2vec2-base-960h** | Secondary ASR + 768-dim embedding extraction. AudioHateXplain shows wav2vec2 embeddings outperform cascaded ASR → text classification. Embeddings stored, ready for a downstream classifier. |
| **YAMNet sound events** | AudioSet-trained, detects gunshots/explosions/screaming. Non-decisive boost — contributes to fusion but cannot trigger alerts alone. |
| **IncidentState with EMA** | Cross-turn score accumulation + consecutive-rise escalation. Inspired by SafeSpeech's conversation-level analysis — individual turns may be benign but escalation patterns are not. |
| **Speculative parallel LLM** | Second opinion fired speculatively in parallel, consumed only when the primary detector returns UNCERTAIN. Adds latency only when it matters. |
| **Continuous features, not binary** | RMS and spectral centroid normalized to [0,1] and used as continuous signals. A slightly raised voice contributes proportionally, not the same as shouting. |
| **Pydantic models everywhere** | Type-safe contracts between engines. Same schemas for validation and serialization. |
| **API resilience** | All OpenAI calls use `max_retries=3` with exponential backoff and 30s timeout. Configurable via settings. |

## Fusion logic

```
effective_threat = threat_score × asr_confidence

final_score = 0.6 × effective_threat
            + 0.2 × rms_normalized
            + 0.2 × keyword_boost
            + directed_boost × confidence_in_direction
            + weight_text_classifier × toxicity_score
            + yamnet_weight × threat_event_confidence

Thresholds:
  < 0.4           → NO_ACTION
  0.4 – 0.6       → LOG
  0.6 – 0.75      → UNCERTAIN (triggers speculative LLM)
  0.7 – 0.9       → ALERT
  ≥ 0.9           → ESCALATE

Escalation boost: if IncidentState.is_escalating → promote one tier
Cooldown: suppress duplicate alerts from same source within 30s window
```

All weights and thresholds are configurable in `config.py` (marked `# TUNABLE`). They are educated guesses — with a labelled dataset, ROC/PR analysis and cross-validation would replace them with empirically justified values. See [ANALYSIS.md, Section 4](ANALYSIS.md#4-the-calibration-gap) for details.

## Relevant research

Three papers informed the architecture and are included in `relevant_research/` for reference:

| Paper | Key insight applied |
|---|---|
| **An Investigation Into Explainable Audio Hate Speech Detection** (AudioHateXplain) | wav2vec2 embeddings outperform cascaded ASR → text classification. End-to-end audio features capture paralinguistic cues lost in transcription. |
| **Multimodal Audio Violence Detection: Fusion of Acoustic Signals and Semantics** | Dual-branch (audio + text) with XGBoost meta-learner outperforms any fixed-weight fusion. The relative reliability of each modality varies by input type. |
| **SafeSpeech: A Comprehensive and Interactive Tool for Analysing Sexist and Abusive Language in Conversations** | Conversation-level analysis catches escalation patterns invisible at the utterance level. Speaker profiling across turns reveals cumulative threat. |

## Setup

```bash
cd challenge/
pip install -e ".[dev]"
```

Create `.env` with your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

## Usage

Process all test files:
```bash
threat-pipeline run-all --audio-dir audio/
```

Process a single file:
```bash
threat-pipeline run --file audio/keyword_only.wav
```

JSON output:
```bash
threat-pipeline run --file audio/keyword_only.wav --json
```

Benchmark (10 runs, averaged with standard deviation):
```bash
threat-pipeline benchmark --audio-dir audio/ --runs 10 --output-dir benchmark_results/
```

The benchmark generates four charts: final score per file (avg + stddev error bars), latency per engine, GPT-4o-mini vs DistilRoBERTa comparison, and latency distribution histogram.

## Testing

```bash
pytest                    # full suite (needs OPENAI_API_KEY)
pytest -m "not api"       # offline tests only (no API key needed)
```

| Layer | What it covers | API needed? |
|---|---|---|
| Unit | Models, thresholds, EMA math, event bus, feature extraction | No |
| Integration | AudioIngest + VAD + Features on real WAVs | No |
| Mocked | All engines with mocked API/model responses | No |
| API | Transcription + ThreatDetector with real OpenAI calls | Yes |
| E2E | Full pipeline on all 5 WAV files | Yes |

118 tests total. All engines tested individually and through end-to-end integration.

## Reading the output

When you run `threat-pipeline run --file audio/keyword_only.wav`, the output shows one block per detected speech turn. Here is what to look for:

### Per-turn output

```
--- Turn 0 [0.00s - 2.34s] ---
Transcript: "Give me your phone or I will hurt you"
ASR confidence: 0.82  no_speech_prob: 0.03
Audio: RMS=-14.2dB  centroid=3412Hz  loud=True  sharp=True
Audio (continuous): rms_norm=0.87  centroid_norm=0.71  bandwidth=2100Hz
Threat: high (score=0.88, directed=True, confidence_in_direction=0.92)
Reasoning: Direct physical threat with explicit intent to harm, directed at someone present
Keywords: hurt, give me
Text classifier: toxic (toxicity=0.91, confidence=0.94)
Wav2Vec2 transcript: "GIVE ME YOUR PHONE OR I WILL HURT YOU"
Incident: accumulated=0.88  rising=1  turns=1
Decision: ALERT (final_score=0.83, incident=inc-...)
** ALERT PUBLISHED ** incident=inc-...
```

### Key fields to watch

| Field | What it tells you | What to look for |
|---|---|---|
| **ASR confidence** | How reliable the transcript is (0–1). Derived from Whisper's avg_logprob. | Below 0.5 means the transcript is likely inaccurate — the pipeline automatically discounts the threat score via `threat_score × asr_confidence`. |
| **no_speech_prob** | Whisper's estimate that the segment contains no speech. | Above 0.5 triggers the empty-transcript guard. High values on speech segments suggest Whisper is struggling with the audio. |
| **Threat score + level** | GPT-4o-mini's semantic assessment (0–1, none/low/medium/high/critical). | The raw LLM opinion before fusion. Compare with final_score to see how audio features and other signals modify it. |
| **directed + confidence_in_direction** | Is the threat aimed at someone present? How sure is the LLM? | `directed=True` with high confidence adds a boost. `directed=False` means the LLM thinks it is TV, quoting, or self-directed speech. |
| **Reasoning** | The LLM's explanation for its assessment. | Read this to understand *why* the score is what it is. Helps catch false positives (e.g., "figurative expression, not a real threat"). |
| **Text classifier** | DistilRoBERTa toxicity label and score. | Currently weight=0.0 in fusion (pending calibration), but the output shows what the local model thinks. Useful for comparing against the LLM. |
| **Wav2Vec2 transcript** | Secondary ASR from wav2vec2 CTC decoder. | Compare with Whisper transcript — disagreement suggests unreliable ASR. All-caps is normal for CTC output. |
| **Audio (continuous)** | rms_norm and centroid_norm (0–1). | These feed directly into fusion. High rms_norm = loud. High centroid_norm = sharp/strained voice. |
| **Incident** | Cross-turn state: accumulated EMA score, consecutive rising turns, total turn count. | `rising=3+` triggers escalation boost. Watch accumulated score to see if the situation is building over multiple turns. |
| **Decision** | The final action after fusion: NO_ACTION, LOG, UNCERTAIN, ALERT, or ESCALATE. | This is the bottom line. `final_score` is what the fusion formula produced. `(suppressed)` means cooldown prevented a duplicate alert. |

### Summary table (run-all)

When processing multiple files with `run-all`, a summary table at the end shows:

```
File                           Turns Alerts Action       Score  Latency
------------------------------ ----- ------ ------------ ------ --------
casual_chat.wav                    2      0 no_action      0.18   2.341s
keyword_only.wav                   1      1 escalate       0.91   1.876s
```

The **Score** column is the maximum `final_score` across all turns in that file. **Action** is the highest-severity action triggered. **Latency** is wall-clock time for the full file.

## Expected results

| File | Expected action | Why |
|---|---|---|
| `casual_chat.wav` | NO_ACTION | Benign content, normal loudness |
| `heated_argument.wav` | LOG or ALERT | Emotional escalation, raised voice, no direct threat words |
| `keyword_only.wav` | ALERT or ESCALATE | Explicit threat directed at a person |
| `false_positive_tv.wav` | NO_ACTION or LOG | Angry tone but not directed at anyone present |
| `muffled_noise.wav` | NO_ACTION | Poor audio quality, ASR confidence low → dampens score |

## Project structure

```
challenge/
├── src/threat_pipeline/
│   ├── cli.py                 # CLI entry point (run, run-all, benchmark)
│   ├── pipeline.py            # Orchestrator — parallel execution groups
│   ├── config.py              # All tunable parameters (Pydantic Settings)
│   ├── models.py              # Data contracts (Pydantic)
│   ├── engine_base.py         # Abstract Engine[TIn, TOut] with timing
│   ├── event_bus.py           # In-process pub/sub
│   ├── incident_state.py      # Cross-turn EMA + escalation detection
│   └── engines/
│       ├── audio_ingestion.py # WAV load + resample
│       ├── vad.py             # Silero VAD segmentation
│       ├── transcription.py   # Whisper API (verbose_json)
│       ├── audio_features.py  # librosa: RMS, MFCCs, spectral, mel
│       ├── threat_detector.py # GPT-4o-mini structured outputs
│       ├── text_classifier_engine.py  # DistilRoBERTa toxicity
│       ├── wav2vec2_engine.py         # CTC transcript + 768-dim embedding
│       ├── yamnet_engine.py           # Google YAMNet sound events
│       ├── speculative_detector.py    # Secondary LLM for UNCERTAIN
│       ├── alert_decision.py  # Weighted fusion → action
│       └── alert_publisher.py # EventBus publishing
├── tests/                     # 118 tests (unit, integration, mocked, API, E2E)
├── audio/                     # 5 provided WAV files
├── relevant_research/         # 3 papers informing the architecture
├── ANALYSIS.md                # Design rationale, weaknesses, calibration roadmap
└── challenge.md               # Original problem statement
```
