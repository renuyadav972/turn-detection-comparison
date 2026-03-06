# Turn Detection Comparison Demo

A Pipecat voice agent that lets you compare four turn detection modes side-by-side over real phone calls, then visualize the results in a metrics dashboard.

## Turn Detection Modes

| Mode | How it works |
|------|-------------|
| **Smart Turn** | Combines silence detection, semantic analysis, and conversational context using a local ML model |
| **Flux Semantic** | Uses Deepgram's cloud-based semantic endpointing to judge if a sentence feels complete |
| **VAD Only (300ms)** | Voice Activity Detection with a 0.3s silence threshold — fast but aggressive |
| **VAD Only (700ms)** | Voice Activity Detection with a 0.7s silence threshold — safer but slower |

## What's Inside

- `bot.py` — Single-mode voice agent (set mode via `.env` or env var)
- `multi_bot.py` — Runs all modes simultaneously with path-based routing
- `metrics_observer.py` — Captures per-turn pipeline metrics (STT/LLM/TTS latency, interruptions, dead air)
- `dashboard.py` — FastAPI server with a live metrics dashboard
- `static/dashboard.html` — Dark-themed Chart.js UI with Live and Compare views
- `scripts/export.py` — Bakes sessions into a self-contained HTML report

## Setup

```bash
cd turn-detection-demo
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API keys:
- **Deepgram** — STT
- **Google Gemini** — LLM
- **Cartesia** — TTS
- **Plivo** — Telephony

## Running

### Single mode

```bash
# Start ngrok
ngrok http 8000

# Run the bot
TURN_DETECTION_MODE=smart_turn python bot.py -t plivo -x <ngrok-host> --port 8000

# In another terminal, start the dashboard
python dashboard.py
```

### All modes at once

```bash
python multi_bot.py --proxy <ngrok-host> --port 8000
```

### Trigger a call

Use the Plivo REST API or the helper script in `scripts/` to initiate an outbound call with `answer_url` pointing to your ngrok tunnel.

### Export results

```bash
python scripts/export.py data/sessions/*.json -o comparison.html
```

## Stack

- [Pipecat](https://github.com/pipecat-ai/pipecat) — Voice agent framework
- Deepgram — Speech-to-text
- Google Gemini — LLM
- Cartesia — Text-to-speech
- Plivo — Telephony
- FastAPI + Chart.js — Dashboard
