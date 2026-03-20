# eNOugh AI Wearable — Threat Detection Challenge

## The Task

Your mission isn't just to build code — it's to prototype a critical slice of what makes eNOugh's AI wearables work in real life.

Every second counts when someone is in a tense situation. The system must **hear, understand, reason, and act** in near-real time: detecting threat signals from short bursts of audio, explaining *why* it believes something is dangerous, and producing a structured alert that can immediately be handled by our **Alarm Receiving Centre (ARC)** and response operators.

Build a small service that:

1. Listens to audio like a live wearable
2. Splits it into short spoken "turns" (using VAD / silence detection)
3. Transcribes each turn with Whisper
4. Uses an LLM (with schema-validated JSON output) plus deterministic audio/text signals to decide: *"Should we raise an alert?"*
5. If yes — publishes a structured alert event (with reasoning and scores)
6. Measures latency per step

---

## Prerequisites

- **OpenAI API key**
- **Audio corpus** — 5 WAV files provided:

| File | Purpose |
|---|---|
| `casual_chat.wav` | Benign baseline |
| `heated_argument.wav` | Escalating conflict |
| `keyword_only.wav` | Clear threat phrase |
| `false_positive_tv.wav` | Angry TV / non-directed speech |
| `muffled_noise.wav` | Hard ASR scenario |

---

## OpenAI API Notes (recommendation only)

- **Speech-to-text:** use the Audio API transcriptions endpoint (e.g. `whisper-1`), or newer transcription snapshots.
- **File upload limit:** 25 MB; common formats like WAV are supported.
- **Structured outputs:** prefer **Structured Outputs (JSON schema)** so detector results are machine-safe.

> Candidates are welcome to surprise us with better models, predictable formats, or anything else they deem needed.

---

## Steps to Complete

### 1. Set Up Audio Ingestion

- Take the provided WAV files and stream them as if coming from a live wearable device.
- Segment the audio into discrete "turns" representing short bursts of speech.

### 2. Transcribe Speech

- Use the OpenAI API to convert each completed audio turn into text.
- Add a formatting step that cleans up the raw transcription for downstream reasoning.
- Ensure this step produces structured, machine-readable output.

### 3. Design and Implement Threat Detection

- Decide what constitutes a "threat signal" in audio and text.
- Combine **deterministic heuristics** with **LLM-based reasoning** to assess whether a given turn is concerning.
- Make your detection **explainable** and **structured**.

### 4. Incorporate Audio-Level Signals

- Extract at least one signal directly from the audio (e.g. intensity, loudness, abrupt changes).
- Use it as an additional input to your threat decision.

### 5. Fuse Signals and Decide When to Alert

- Combine text, audio, and LLM outputs into a single decision.
- Define thresholds and logic that determine when an alert should fire.

### 6. Handle Incident-Level Behavior

- Ensure your system doesn't spam alerts for the same ongoing situation.
- Introduce basic **incident or cooldown logic** to group related turns.

### 7. Publish Alerts

- Spin up a real-time message streaming service.
- Design an alert event schema and publish alerts.
- Include enough context for a downstream human operator or system to understand *why* the alert fired.

### 8. Create a Sandbox Environment

- Provide a simple CLI or script to run the pipeline on sample audio.
- Include minimal automated checks (tests) to show the system behaves as expected.

---

## Tips

| # | Guidance |
|---|---|
| 1 | **Focus on the AI pipeline, not the UI.** A CLI is more than enough. We want to see your audio handling, reasoning, and alerting logic. |
| 2 | **Make your decisions explicit.** You won't be given a keyword list, threat taxonomy, or alert schema. Design your own and explain *why* in your README. |
| 3 | **Keep latency in mind.** This is a real-time safety system. Even if your implementation is simple, show awareness of where time is spent and what would matter in production. |
