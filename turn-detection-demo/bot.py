"""
Turn Detection Comparison Demo
==============================
A single Pipecat voice agent with a config toggle between four turn detection
approaches:

  1. smart_turn    — Silero VAD + Smart Turn v3 local model + Deepgram endpointing
  2. flux_semantic — Deepgram Flux native semantic endpointing (cloud)
  3. vad_only      — Fixed silence timeout (e.g. 300ms or 700ms)

Run:
    TURN_DETECTION_MODE=smart_turn   python bot.py -t plivo -x <ngrok-host> --port 8000
    TURN_DETECTION_MODE=flux_semantic python bot.py -t plivo -x <ngrok-host> --port 8000
    TURN_DETECTION_MODE=vad_only     python bot.py -t plivo -x <ngrok-host> --port 8000
"""

import asyncio
import os
import uuid
from enum import Enum

import httpx

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from metrics_observer import MetricsCollectorObserver

load_dotenv(override=False)  # Shell env vars take precedence over .env


# ---------------------------------------------------------------------------
# Turn detection toggle
# ---------------------------------------------------------------------------


class TurnDetectionMode(str, Enum):
    VAD_ONLY = "vad_only"
    SMART_TURN = "smart_turn"
    FLUX_SEMANTIC = "flux_semantic"
    # Keep legacy alias so old .env files still work
    SEMANTIC = "semantic"


def _resolve_mode(mode: TurnDetectionMode) -> TurnDetectionMode:
    """Map legacy 'semantic' to 'smart_turn'."""
    if mode is TurnDetectionMode.SEMANTIC:
        return TurnDetectionMode.SMART_TURN
    return mode


def create_turn_strategies(mode: TurnDetectionMode, vad_stop_secs: float | None = None) -> UserTurnStrategies | None:
    """Return the appropriate UserTurnStrategies for the chosen mode."""
    if mode is TurnDetectionMode.VAD_ONLY:
        timeout = vad_stop_secs if vad_stop_secs is not None else float(os.getenv("VAD_STOP_SECS", "0.7"))
        logger.info(f"Turn detection: VAD-only (silence timeout {timeout}s)")
        return UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=timeout)],
        )

    if mode is TurnDetectionMode.SMART_TURN:
        stop_secs = float(os.getenv("SMART_TURN_STOP_SECS", "3.0"))
        logger.info(f"Turn detection: Smart Turn v3 (stop_secs={stop_secs})")
        return UserTurnStrategies(
            stop=[
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(
                        params=SmartTurnParams(stop_secs=stop_secs),
                    ),
                )
            ],
        )

    if mode is TurnDetectionMode.FLUX_SEMANTIC:
        logger.info("Turn detection: Deepgram Flux semantic (native EOT)")
        # Flux handles turn detection natively — no pipecat turn strategy needed
        return None

    return None


def create_stt(mode: TurnDetectionMode):
    """Create the appropriate STT service for the mode."""
    if mode is TurnDetectionMode.FLUX_SEMANTIC:
        from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

        eot_threshold = float(os.getenv("FLUX_EOT_THRESHOLD", "0.7"))
        eot_timeout_ms = int(os.getenv("FLUX_EOT_TIMEOUT_MS", "5000"))
        eager_eot = os.getenv("FLUX_EAGER_EOT_THRESHOLD")
        params = DeepgramFluxSTTService.InputParams(
            eot_threshold=eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
        )
        if eager_eot is not None:
            params.eager_eot_threshold = float(eager_eot)
        logger.info(
            f"STT: Deepgram Flux (eot_threshold={eot_threshold}, "
            f"eot_timeout_ms={eot_timeout_ms}, eager_eot={eager_eot})"
        )
        return DeepgramFluxSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="flux-general-en",
            params=params,
        )

    # Regular Deepgram for Smart Turn and VAD modes
    logger.info("STT: Deepgram Nova 3")
    return DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), model="nova-3")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a personal AI assistant on a phone call — like Siri or Alexa, "
    "but over the phone. The caller will ask you questions about weather, "
    "reminders, conversions, general knowledge, etc. "
    "Keep your responses to 1-2 short sentences. "
    "Your output will be spoken aloud, so avoid special characters, "
    "markdown, or lists. Be conversational and natural."
)


async def run_bot(transport: BaseTransport, handle_sigint: bool, mode=None, vad_stop_secs=None, call_id=None):
    if mode is None:
        mode = TurnDetectionMode(os.getenv("TURN_DETECTION_MODE", "smart_turn"))
    mode = _resolve_mode(mode)
    if vad_stop_secs is None:
        vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "0.7"))

    turn_strategies = create_turn_strategies(mode, vad_stop_secs=vad_stop_secs)
    stt = create_stt(mode)

    # Metrics collection
    session_id = str(uuid.uuid4())
    data_dir = os.path.join(os.path.dirname(__file__), "data", "sessions")
    smart_stop = float(os.getenv("SMART_TURN_STOP_SECS", "3.0"))
    metrics_observer = MetricsCollectorObserver(
        session_id=session_id,
        mode=mode.value,
        config={"vad_stop_secs": vad_stop_secs, "smart_turn_stop_secs": smart_stop},
        data_dir=data_dir,
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        model="sonic-3",
        sample_rate=16000,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)

    # For Flux mode, skip Silero VAD — Flux handles speech detection natively
    use_vad = mode is not TurnDetectionMode.FLUX_SEMANTIC
    user_params = LLMUserAggregatorParams(
        user_turn_strategies=turn_strategies,
        **({"vad_analyzer": SileroVADAnalyzer()} if use_vad else {}),
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=user_params,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
            observers=[metrics_observer],
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start Plivo call recording (fire-and-forget)
        if call_id:
            asyncio.create_task(_start_recording(call_id))

        logger.info("Client connected — sending greeting")
        await task.queue_frames([TTSSpeakFrame(text="Hey! What can I help you with?")])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        # Fetch Plivo recording URL and save to session
        if call_id:
            asyncio.create_task(_fetch_recording(call_id, session_id, data_dir))
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def _start_recording(call_id: str):
    """Fire-and-forget: start Plivo call recording via REST API."""
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"https://api.plivo.com/v1/Account/{auth_id}/Call/{call_id}/Record/",
                auth=(auth_id, auth_token),
                json={"time_limit": 3600, "file_format": "mp3"},
            )
            logger.info(f"Plivo recording started: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to start Plivo recording: {e}")


async def _fetch_recording(call_id: str, session_id: str, data_dir: str):
    """Poll Plivo API for the call recording URL and update the session JSON."""
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return

    import json

    # Plivo needs time to process the recording
    for attempt in range(12):
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"https://api.plivo.com/v1/Account/{auth_id}/Recording/",
                    auth=(auth_id, auth_token),
                    params={"call_uuid": call_id, "limit": 1},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    objects = data.get("objects", [])
                    if objects:
                        recording_url = objects[0].get("recording_url")
                        if recording_url:
                            logger.info(f"Plivo recording ready: {recording_url}")
                            # Update session JSON with recording URL
                            session_path = os.path.join(data_dir, f"{session_id}.json")
                            if os.path.exists(session_path):
                                with open(session_path) as f:
                                    session = json.load(f)
                                session["recording_url"] = recording_url
                                with open(session_path, "w") as f:
                                    json.dump(session, f, indent=2)
                                logger.info(f"Recording URL saved to session {session_id}")
                            return
        except Exception as e:
            logger.warning(f"Recording fetch attempt {attempt + 1}: {e}")

    logger.warning(f"Could not fetch recording for call {call_id} after 60s")


# ---------------------------------------------------------------------------
# Entry point — Pipecat runner discovers this function
# ---------------------------------------------------------------------------


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud and the dev runner."""
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, call_id=call_data["call_id"])


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
