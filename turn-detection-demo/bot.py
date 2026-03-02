"""
Turn Detection Comparison Demo
==============================
A single Pipecat voice agent with a config toggle between two turn detection
approaches — VAD-only (fixed 700ms silence) vs semantic (Smart Turn v3).

Run:
    TURN_DETECTION_MODE=semantic python bot.py -t plivo -x <ngrok-host> --port 8000
    TURN_DETECTION_MODE=vad_only python bot.py -t plivo -x <ngrok-host> --port 8000
"""

import os
import uuid
from enum import Enum

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
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

load_dotenv()

# ---------------------------------------------------------------------------
# Turn detection toggle
# ---------------------------------------------------------------------------


class TurnDetectionMode(str, Enum):
    VAD_ONLY = "vad_only"
    SEMANTIC = "semantic"


def create_turn_strategies(mode: TurnDetectionMode) -> UserTurnStrategies:
    """Return the appropriate UserTurnStrategies for the chosen mode."""
    if mode is TurnDetectionMode.VAD_ONLY:
        timeout = float(os.getenv("VAD_STOP_SECS", "0.7"))
        logger.info(f"Turn detection: VAD-only (silence timeout {timeout}s)")
        return UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=timeout)],
        )

    # Semantic mode — Smart Turn v3
    stop_secs = float(os.getenv("SMART_TURN_STOP_SECS", "3.0"))
    logger.info(f"Turn detection: Semantic / Smart Turn v3 (stop_secs={stop_secs})")
    return UserTurnStrategies(
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(stop_secs=stop_secs),
                ),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a friendly, helpful voice assistant on a phone call. "
    "Keep your responses to 1-2 short sentences. "
    "Your output will be spoken aloud, so avoid special characters, "
    "markdown, or lists. Be conversational and natural."
)


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    mode = TurnDetectionMode(os.getenv("TURN_DETECTION_MODE", "semantic"))
    turn_strategies = create_turn_strategies(mode)

    # Metrics collection
    session_id = str(uuid.uuid4())
    data_dir = os.path.join(os.path.dirname(__file__), "data", "sessions")
    vad_stop = float(os.getenv("VAD_STOP_SECS", "0.7"))
    smart_stop = float(os.getenv("SMART_TURN_STOP_SECS", "3.0"))
    metrics_observer = MetricsCollectorObserver(
        session_id=session_id,
        mode=mode.value,
        config={"vad_stop_secs": vad_stop, "smart_turn_stop_secs": smart_stop},
        data_dir=data_dir,
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-5-20250514",
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
        model="eleven_turbo_v2_5",
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=turn_strategies,
            vad_analyzer=SileroVADAnalyzer(),
        ),
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
        logger.info("Client connected — sending greeting")
        messages.append(
            {
                "role": "system",
                "content": "Greet the caller warmly and ask how you can help.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


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

    await run_bot(transport, runner_args.handle_sigint)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
