"""
Unified server for Railway deployment.

Combines the Pipecat voice bot (Plivo webhook + WebSocket) with the metrics
dashboard into a single FastAPI process.

    POST /      — Plivo answer_url webhook (returns Stream XML)
    WS   /ws    — Pipecat bot WebSocket (Plivo connects here)
    GET  /      — Dashboard UI
    GET  /api/* — Dashboard REST API
    GET  /health — Railway health check
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from loguru import logger

load_dotenv(override=False)

app = FastAPI(title="Turn Detection Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN", os.getenv("PUBLIC_DOMAIN", "localhost:8000"))

# ---------------------------------------------------------------------------
# Plivo webhook — must be registered before dashboard's GET /
# ---------------------------------------------------------------------------

PLIVO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://{domain}/ws</Stream>
</Response>"""


@app.post("/")
async def plivo_webhook(request: Request):
    """Plivo answer_url — tells Plivo to open a bidirectional audio stream."""
    domain = request.headers.get("host") or DOMAIN
    xml = PLIVO_XML.format(domain=domain)
    logger.info(f"Plivo webhook hit — streaming to wss://{domain}/ws")
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Bot WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accept Plivo audio stream and run the Pipecat bot."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    from bot import bot
    from pipecat.runner.types import WebSocketRunnerArguments

    args = WebSocketRunnerArguments(websocket=websocket)
    await bot(args)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dashboard routes (GET /, /api/*, /static/*, /recordings/*)
# ---------------------------------------------------------------------------

from dashboard import register_dashboard  # noqa: E402

register_dashboard(app)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port} (domain={DOMAIN})")
    uvicorn.run(app, host="0.0.0.0", port=port)
