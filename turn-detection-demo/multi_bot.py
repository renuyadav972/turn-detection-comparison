"""
Multi-Mode Turn Detection Server
=================================
Single FastAPI app serving 3 turn detection modes simultaneously via
path-based routing. Each mode gets its own Plivo webhook + WebSocket path.

    python multi_bot.py --proxy <ngrok-host> --port 8000

Routes:
    POST /{mode}/     → Plivo XML with WebSocket URL
    WS   /{mode}/ws   → WebSocket handler for audio streaming
    GET  /            → Status endpoint listing all modes
"""

import argparse
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from bot import TurnDetectionMode, run_bot

load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Mode configurations
# ---------------------------------------------------------------------------

MODE_CONFIGS = {
    "smart_turn": {
        "mode": TurnDetectionMode.SMART_TURN,
        "vad_stop_secs": 0.7,
        "label": "Smart Turn + Semantic",
    },
    "flux": {
        "mode": TurnDetectionMode.FLUX_SEMANTIC,
        "vad_stop_secs": 0.7,
        "label": "Deepgram Flux Semantic",
    },
    "vad300": {
        "mode": TurnDetectionMode.VAD_ONLY,
        "vad_stop_secs": 0.3,
        "label": "VAD 300ms",
    },
    "vad700": {
        "mode": TurnDetectionMode.VAD_ONLY,
        "vad_stop_secs": 0.7,
        "label": "VAD 700ms",
    },
}


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_mode_routes(app: FastAPI, key: str, config: dict, proxy: str):
    """Register webhook + websocket routes for a single mode.

    Uses a factory function to avoid the closure-in-loop capture bug.
    """

    @app.post(f"/{key}/")
    async def webhook():
        logger.info(f"Plivo webhook hit for mode: {config['label']}")
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            '  <Stream bidirectional="true" keepCallAlive="true"'
            f' contentType="audio/x-mulaw;rate=8000">wss://{proxy}/{key}/ws</Stream>\n'
            "</Response>"
        )
        return HTMLResponse(content=xml, media_type="application/xml")

    @app.websocket(f"/{key}/ws")
    async def websocket_handler(websocket: WebSocket):
        await websocket.accept()
        logger.info(f"WebSocket connected for mode: {config['label']}")

        transport_type, call_data = await parse_telephony_websocket(websocket)
        logger.info(f"Transport: {transport_type}, mode: {config['label']}")

        serializer = PlivoFrameSerializer(
            stream_id=call_data["stream_id"],
            call_id=call_data["call_id"],
            auth_id=os.getenv("PLIVO_AUTH_ID", ""),
            auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
        )

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=serializer,
            ),
        )

        await run_bot(
            transport,
            handle_sigint=False,
            mode=config["mode"],
            vad_stop_secs=config["vad_stop_secs"],
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(proxy: str) -> FastAPI:
    app = FastAPI(title="Multi-Mode Turn Detection Server")

    for key, config in MODE_CONFIGS.items():
        register_mode_routes(app, key, config, proxy)
        logger.info(f"Registered mode: {config['label']} at /{key}/")

    @app.get("/")
    async def status():
        modes = {
            key: {
                "label": cfg["label"],
                "webhook": f"https://{proxy}/{key}/",
                "websocket": f"wss://{proxy}/{key}/ws",
            }
            for key, cfg in MODE_CONFIGS.items()
        }
        return JSONResponse({"status": "running", "modes": modes})

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-mode turn detection server")
    parser.add_argument("--proxy", "-x", required=True, help="ngrok hostname")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.proxy)
    uvicorn.run(app, host=args.host, port=args.port)
