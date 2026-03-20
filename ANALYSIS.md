# Analysis

This document is the single reference for understanding our design rationale, the weaknesses we found during testing, and where the pipeline would go next with proper training data.

---

## Index

1. [Why this architecture](#1-why-this-architecture)
2. [How signals are fused](#2-how-signals-are-fused)
3. [What we tested and what broke](#3-what-we-tested-and-what-broke)
4. [The calibration gap](#4-the-calibration-gap)
5. [What a training dataset would unlock](#5-what-a-training-dataset-would-unlock)
6. [Production path](#6-production-path)
7. [Weakness log](#7-weakness-log)

---

## 1. Why this architecture

The pipeline follows a **multi-engine fusion** approach rather than relying on a single model. Audio comes in, gets segmented into speech turns by Silero VAD, and then multiple engines process each turn in parallel: Whisper transcribes, librosa extracts acoustic features, wav2vec2 produces embeddings, YAMNet listens for non-speech sounds (gunshots, screaming), GPT-4o-mini reasons about threat semantics, and DistilRoBERTa classifies toxicity.

This is not accidental. The research consistently shows that **no single modality is sufficient** for violence and threat detection:

- **AudioHateXplain** (Ghosh et al.) demonstrates that cascading ASR into a text classifier compounds errors — ASR mistakes propagate into wrong classifications. Their wav2vec2-based end-to-end approach outperforms the cascade, specifically because it bypasses the ASR bottleneck. This is why we extract wav2vec2 embeddings as a parallel signal rather than depending entirely on Whisper.

- **Multimodal Audio Violence Detection** (acoustic + semantic fusion) shows that a dual-branch architecture — one branch on audio features (Random Forest on MFCCs, spectral contrast, mel spectrograms), another on text (fine-tuned BERT) — with learned fusion via XGBoost significantly outperforms any single branch. The key insight: **the relative reliability of each modality varies by input type**. A fixed-weight combination cannot capture this, but a learned meta-learner can.

- **SafeSpeech** (conversation-level abuse detection) highlights that **utterance-level classification misses escalation patterns**. Threats often build across multiple turns — individually benign sentences that become threatening in sequence. This motivated our IncidentState with EMA-based cross-turn score accumulation.

These three papers are included in `relevant_research/` for reference.

### What we built vs. what the research suggests

| Research insight | Our implementation | Gap |
|---|---|---|
| Multi-branch fusion outperforms single-path | 6 parallel engines feeding fusion layer | Fusion weights are hand-tuned, not learned |
| ASR errors propagate — use audio embeddings directly | wav2vec2 extracts 768-dim embeddings | Embeddings stored but not yet consumed by a classifier |
| Toxicity classifier provides fast local signal | DistilRoBERTa loaded and running | Weight set to 0.0 — needs benchmarking to justify inclusion |
| Conversation-level analysis catches escalation | IncidentState with EMA + consecutive-rise detection | No speaker diarization — cannot attribute who is escalating |
| Non-speech violence (gunshots, impacts) matters | YAMNet detects AudioSet threat classes | Only boosts score; VAD still filters most non-speech |

The architecture is designed so that **every component is wired in and producing output**, even where we could not complete the calibration. With a labelled dataset, each of these gaps becomes a straightforward data science task — not an engineering rewrite.

---

## 2. How signals are fused

The fusion formula combines six signals into a single score:

```
effective_threat = threat_score × asr_confidence

final_score = 0.6 × effective_threat
            + 0.2 × rms_normalized
            + 0.2 × keyword_boost
            + directed_boost × confidence_in_direction
            + weight_text_classifier × toxicity_score
            + yamnet_weight × threat_event_confidence
```

Multiplying `threat_score × asr_confidence` means the LLM's assessment is automatically discounted when Whisper is uncertain — a noisy transcript produces a lower effective signal regardless of what the LLM says.

The continuous features (RMS normalized to [0,1], spectral centroid normalized) replace the original binary is_loud/is_sharp flags, preserving magnitude information. A slightly raised voice contributes proportionally rather than being treated the same as shouting.

### Decision thresholds

| Range | Action | Rationale |
|---|---|---|
| < 0.4 | NO_ACTION | Low combined evidence |
| 0.4 – 0.6 | LOG | Worth recording, not worth alerting |
| 0.6 – 0.75 | UNCERTAIN | Borderline — triggers speculative second LLM opinion |
| 0.7 – 0.9 | ALERT | Sufficient evidence for operator notification |
| ≥ 0.9 | ESCALATE | High-confidence threat, immediate response |

The UNCERTAIN band is important: rather than making a binary call on borderline scores, the pipeline fires a parallel LLM with a different prompt and higher temperature, then takes the more cautious of the two opinions. This costs API time only when needed (the speculative call runs in parallel, already started before we know whether we need it).

### Cross-turn escalation

IncidentState tracks an EMA (exponential moving average) of threat scores across turns for each audio source. If three or more consecutive turns show rising scores, the pipeline promotes the alert tier. This catches gradual escalation patterns that no single turn would trigger — directly addressing the SafeSpeech insight about conversation-level dynamics.

### Honest assessment

These weights (0.6, 0.2, 0.2) and thresholds (0.4, 0.7, 0.9) are **educated guesses**. They produce reasonable results on our test audio, but they have not been validated with ROC/PR analysis or optimized on labelled data. Section 4 explains exactly what calibration would look like.

---

## 3. What we tested and what broke

We tested with two audio sets: the 5 provided WAV files (synthetic/TTS) and a set of real recordings with accented English (`gerva_recordings/`). The real recordings exposed several weaknesses that the synthetic audio did not.

### ASR breaks on accented speech

Whisper's language auto-detection classified accented English as Spanish or Portuguese. **Fixed** by hardcoding `whisper_language="en"`. Even after fixing, ASR confidence dropped to 0.44–0.67 on accented speech versus ~0.85+ on synthetic audio. The pipeline handles this gracefully through the `threat_score × asr_confidence` formula, but it means accented speakers get systematically lower threat scores for equivalent utterances.

### VAD over-segments accented speech

"Step back right now" was split into ["Step", "back", "right now"]. Each fragment was scored independently as benign — the threat only makes sense as a complete sentence. This is the most impactful weakness: it breaks the semantic unit before the LLM ever sees it. The VAD thresholds (0.5 probability, 250ms minimum speech, 100ms minimum silence) were tuned for standard English pronunciation cadence.

### Implicit threats are under-scored

The LLM scored "Shut up and get back over here" at 0.26 and "Do exactly what I say" at 0.38. Both are clearly threatening in context, but the LLM treats them as assertive rather than dangerous. This is a prompt engineering and potentially a fine-tuning problem — the LLM needs examples of coercive control patterns, not just explicit violence.

### What worked well

- The 5 provided WAV files are classified correctly: `casual_chat.wav` → NO_ACTION, `keyword_only.wav` → ALERT/ESCALATE, `false_positive_tv.wav` → NO_ACTION or LOG
- ASR confidence propagation successfully dampens scores when transcripts are unreliable
- The speculative LLM provides useful second opinions on borderline cases
- Per-engine latency tracking makes it clear where time is spent (Whisper and GPT-4o-mini dominate)

---

## 4. The calibration gap

This is the most important section of this analysis. **The pipeline architecture is sound, but its parameters are not empirically validated.** Every weight, threshold, and boost factor was set by reasoning about what should matter, not by measuring what actually discriminates threats from non-threats on real data.

### What calibration requires

**A labelled dataset.** Specifically, 50–200 audio segments annotated with ground-truth actions (NO_ACTION, LOG, ALERT, ESCALATE). With that, standard data science techniques apply:

1. **Threshold selection via ROC/PR curves** — Plot the pipeline's `final_score` against ground truth. Choose thresholds at the desired operating point (e.g., 95% recall at maximum precision). This replaces our guessed 0.4/0.7/0.9 with empirically justified values.

2. **Weight optimization** — Grid search or Bayesian optimization over the fusion weights (threat, loudness, keyword, directed_boost, text_classifier, yamnet). Cross-validate on the labelled set. The current 0.6/0.2/0.2 split may turn out to be reasonable, or it may not — we cannot know without data.

3. **DistilRoBERTa activation** — The text classifier runs but its weight is 0.0 because we have not measured whether it helps or hurts. With labelled data, we would compare pipeline accuracy with and without the toxicity signal, then set the weight accordingly.

4. **wav2vec2 classifier training** — The 768-dimensional embeddings are extracted but unused. A simple linear probe or small MLP trained on threat/non-threat labels would convert these embeddings into an independent audio threat score — a third branch in the fusion, as the Multimodal Audio Violence Detection paper demonstrates.

5. **XGBoost meta-learner** — The aspirational replacement for the weighted sum. Instead of hand-tuning six weights, train a gradient-boosted classifier on `[llm_score, text_classifier_score, audio_features, wav2vec2_score, yamnet_confidence, incident_state]` → action. This learns non-linear interactions (e.g., "high yamnet confidence + low ASR confidence = trust the sound event, not the transcript").

### Why we did not calibrate

We had 5 synthetic WAV files and a handful of real recordings — not enough to train or validate anything. A labelled dataset of this kind would typically come from operational data (recorded incidents with ARC operator annotations) or from a purpose-built corpus. Building one was outside the scope of this challenge, but the pipeline is structured so that plugging one in requires no architectural changes.

---

## 5. What a training dataset would unlock

| Technique | Data needed | What it improves |
|---|---|---|
| ROC/PR threshold calibration | 50+ labelled (score, action) pairs | Alert thresholds — defensible operating point |
| Fusion weight optimization | 100+ labelled samples | Weight balance between engines |
| DistilRoBERTa activation | Same labelled set + A/B comparison | Decides if local toxicity signal helps or hurts |
| wav2vec2 linear probe | 100+ threat/non-threat audio clips | Converts embeddings into usable audio-only threat score |
| XGBoost meta-learner | 200+ labelled samples across all inputs | Replaces hand-tuned fusion with learned non-linear model |
| DistilRoBERTa fine-tuning | 500+ threat-specific examples | Domain-adapted text classifier (vs generic toxicity) |
| VAD threshold tuning | 50+ accented/diverse speech samples | Reduces over-segmentation on non-standard English |
| Prompt optimization for implicit threats | 20–50 coercion/control examples | Fixes under-scoring of indirect threats |

The first four rows are straightforward data science work with standard tooling (sklearn, xgboost, HuggingFace Trainer). The pipeline already outputs all the features these techniques need — it is a matter of having the labels.

---

## 6. Production path

The current pipeline is a prototype that uses cloud APIs and full-size models. In production on a wearable device, every component has an on-device equivalent:

| Component | Prototype (now) | Production (target) |
|---|---|---|
| Audio input | WAV file | BLE/WebSocket chunked PCM stream |
| VAD | Silero PyTorch | Silero ONNX, INT8 quantized |
| Transcription | Whisper API | whisper.cpp on-device |
| Threat detection | GPT-4o-mini API | Fine-tuned ONNX classifier |
| Text classifier | DistilRoBERTa HuggingFace | Quantized ONNX on-device |
| Audio embeddings | wav2vec2-base-960h | Distilled wav2vec2 ONNX |
| Sound events | YAMNet TF Hub | YAMNet TFLite on-device |
| Alert delivery | In-process EventBus | Redis Pub/Sub → ARC console |

The latency target for on-device deployment is **< 200ms wearable-to-alert**. The current prototype runs in 2–4 seconds per file (dominated by API round-trips). On-device inference would eliminate network latency entirely.

### API resilience (implemented)

All OpenAI API calls now use `max_retries=3` with exponential backoff and a `30-second timeout`, configurable via `api_max_retries` and `api_timeout_seconds` in settings. This prevents the pipeline from hanging on slow or failed API responses.

---

## 7. Weakness log

Empirical weaknesses discovered during testing. This is what we know breaks or underperforms, not theoretical concerns.

| # | Category | Weakness | Severity | Status |
|---|----------|----------|----------|--------|
| W-01 | ASR | Language auto-detection fails on accented English | HIGH | **Fixed** — `whisper_language="en"` |
| W-02 | ASR | Confidence drops to 0.44–0.67 on accented speech | MEDIUM | Open |
| W-03 | ASR | Whisper hallucination risk on noisy/muffled audio | MEDIUM | Open |
| W-04 | VAD | Over-segments accented speech into tiny fragments | HIGH | Open |
| W-05 | VAD | Short fragments lose semantic context for LLM | HIGH | Open |
| W-06 | VAD | Thresholds tuned for standard English only | MEDIUM | Open |
| W-07 | Scoring | Implicit threats under-scored (0.26 for "shut up and get back over here") | HIGH | Open |
| W-08 | Scoring | Coercion under-scored (0.38 for "do exactly what I say") | HIGH | Open |
| W-09 | Scoring | Multi-turn threats split by VAD lose cumulative impact | MEDIUM | Open |
| W-10 | Scoring | Directed-boost cliff: +0.15 boolean jump can flip alert tier | MEDIUM | Open |
| W-11 | API | No retry/backoff on API calls | MEDIUM | **Fixed** — `api_max_retries=3` |
| W-12 | API | No timeout on API calls | MEDIUM | **Fixed** — `api_timeout_seconds=30.0` |
| W-13 | API | Speculative LLM doubles cost; benefit unvalidated | LOW | Open |
| W-14 | Calibration | All weights/thresholds are educated guesses | HIGH | Open — needs labelled data |
| W-15 | Calibration | DistilRoBERTa runs but weight=0.0 | LOW | Open — needs A/B comparison |
| W-16 | Calibration | wav2vec2 embedding extracted but unconsumed | LOW | Open — needs classifier training |
| W-17 | Calibration | Keyword list is 12 static words, no synonyms | MEDIUM | Open |
| W-18 | Environment | No offline/on-device fallback | MEDIUM | Open |
| W-19 | Environment | Non-speech threats mostly invisible to VAD | MEDIUM | Open |
| W-20 | Environment | No speaker diarization — TV indistinguishable from live speech | MEDIUM | Open |
| W-21 | Environment | No code-switching support (mixed-language) | LOW | Open |
