"""
Metrics Collector Observer
==========================
BaseObserver subclass that captures per-turn metrics during a Pipecat voice
agent session and writes them as JSON to disk.

Attach to a PipelineTask via PipelineParams(observers=[...]).
"""

import json
import os
import tempfile
import time
from collections import deque
from datetime import datetime, timezone

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import SmartTurnMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.utils.utils import FrameDirection


class MetricsCollectorObserver(BaseObserver):
    """Captures per-turn latency, transcriptions, and Smart Turn confidence.

    Writes a session JSON file to ``data_dir/{session_id}.json`` after every
    finalized turn, using atomic rename for safe concurrent reads.
    """

    def __init__(
        self,
        *,
        session_id: str,
        mode: str,
        config: dict,
        data_dir: str = "data/sessions",
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._session_id = session_id
        self._mode = mode
        self._config = config
        self._data_dir = data_dir
        self._output_path = os.path.join(data_dir, f"{session_id}.json")

        # Frame dedup — bounded set backed by a deque
        self._processed_frames: set = set()
        self._frame_history: deque = deque(maxlen=200)

        # Session state
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._ended_at = None
        self._turns: list[dict] = []

        # Current turn accumulators
        self._user_started_at: float = 0
        self._user_stopped_at: float = 0
        self._bot_started_at: float = 0
        self._user_text_parts: list[str] = []
        self._bot_text_parts: list[str] = []
        self._smart_turn_data: dict | None = None
        self._turn_number: int = 0

        os.makedirs(data_dir, exist_ok=True)
        logger.info(
            f"MetricsCollector: session={session_id} mode={mode} -> {self._output_path}"
        )

    # ------------------------------------------------------------------
    # Dedup helper
    # ------------------------------------------------------------------

    def _is_duplicate(self, frame_id) -> bool:
        if frame_id in self._processed_frames:
            return True
        self._processed_frames.add(frame_id)
        self._frame_history.append(frame_id)
        # Keep set bounded to deque contents
        if len(self._processed_frames) > len(self._frame_history) + 50:
            self._processed_frames = set(self._frame_history)
        return False

    # ------------------------------------------------------------------
    # Observer callback
    # ------------------------------------------------------------------

    async def on_push_frame(self, data: FramePushed):
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        if self._is_duplicate(data.frame.id):
            return

        frame = data.frame

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._on_user_started(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._on_user_stopped(frame)
        elif isinstance(frame, TranscriptionFrame):
            self._on_transcription(frame)
        elif isinstance(frame, LLMTextFrame):
            self._on_llm_text(frame)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._on_bot_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._on_bot_stopped()
        elif isinstance(frame, MetricsFrame):
            self._on_metrics(frame)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._finalize_session()

    # ------------------------------------------------------------------
    # Frame handlers
    # ------------------------------------------------------------------

    def _on_user_started(self, frame: VADUserStartedSpeakingFrame):
        self._user_started_at = frame.timestamp
        self._user_stopped_at = 0
        self._bot_started_at = 0
        self._user_text_parts = []
        self._bot_text_parts = []
        self._smart_turn_data = None

    def _on_user_stopped(self, frame: VADUserStoppedSpeakingFrame):
        # Actual stop time = timestamp - stop_secs (matches Pipecat convention)
        self._user_stopped_at = frame.timestamp - frame.stop_secs

    def _on_transcription(self, frame: TranscriptionFrame):
        if frame.text and frame.text.strip():
            self._user_text_parts.append(frame.text.strip())

    def _on_llm_text(self, frame: LLMTextFrame):
        if frame.text:
            self._bot_text_parts.append(frame.text)

    def _on_bot_started(self):
        now = time.time()
        self._bot_started_at = now
        # If we haven't recorded user_stopped_at, fall back to user_started_at
        if self._user_stopped_at and self._user_started_at:
            latency_ms = round((now - self._user_stopped_at) * 1000)
        else:
            latency_ms = 0

        self._turn_number += 1
        turn = {
            "turn_number": self._turn_number,
            "user_started_at": self._user_started_at or None,
            "user_stopped_at": self._user_stopped_at or None,
            "bot_started_at": now,
            "response_latency_ms": max(latency_ms, 0),
            "user_text": " ".join(self._user_text_parts),
            "bot_text": "",
        }
        if self._smart_turn_data:
            turn["smart_turn"] = self._smart_turn_data
        self._turns.append(turn)
        logger.info(
            f"MetricsCollector: turn {self._turn_number} latency={latency_ms}ms"
        )

    def _on_bot_stopped(self):
        if self._turns:
            self._turns[-1]["bot_text"] = "".join(self._bot_text_parts)
        self._bot_text_parts = []
        self._flush()

    def _on_metrics(self, frame: MetricsFrame):
        for m in frame.data:
            if isinstance(m, SmartTurnMetricsData):
                self._smart_turn_data = {
                    "is_complete": m.is_complete,
                    "probability": round(m.probability, 4),
                    "inference_time_ms": round(m.inference_time_ms, 1),
                    "e2e_processing_time_ms": round(m.e2e_processing_time_ms, 1),
                }

    def _finalize_session(self):
        self._ended_at = datetime.now(timezone.utc).isoformat()
        self._flush()
        logger.info(
            f"MetricsCollector: session {self._session_id} finalized — "
            f"{len(self._turns)} turns"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict:
        latencies = [t["response_latency_ms"] for t in self._turns if t["response_latency_ms"]]
        smart_probs = [
            t["smart_turn"]["probability"]
            for t in self._turns
            if t.get("smart_turn")
        ]
        smart_infer = [
            t["smart_turn"]["inference_time_ms"]
            for t in self._turns
            if t.get("smart_turn")
        ]

        summary = {
            "total_turns": len(self._turns),
            "avg_response_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "min_response_latency_ms": min(latencies) if latencies else 0,
            "max_response_latency_ms": max(latencies) if latencies else 0,
        }
        if smart_probs:
            summary["avg_smart_turn_probability"] = round(
                sum(smart_probs) / len(smart_probs), 4
            )
        if smart_infer:
            summary["avg_smart_turn_inference_ms"] = round(
                sum(smart_infer) / len(smart_infer), 1
            )
        return summary

    def _flush(self):
        session = {
            "session_id": self._session_id,
            "mode": self._mode,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "config": self._config,
            "turns": self._turns,
            "summary": self._build_summary(),
        }

        # Atomic write: temp file + rename
        fd, tmp_path = tempfile.mkstemp(
            dir=self._data_dir, suffix=".tmp", prefix=".session_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(session, f, indent=2)
            os.rename(tmp_path, self._output_path)
        except Exception:
            logger.exception("MetricsCollector: failed to write session JSON")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
