"""
WebSocket Replay Client
=======================
Connects to multi_bot.py WebSocket endpoints, sends pre-recorded caller audio
using the Plivo WebSocket protocol, and records bot responses.

Produces real metrics (from real STT/LLM/TTS services) and audio recordings
with zero manual effort per mode.

Usage:
    python scripts/replay_client.py                          # all 3 modes
    python scripts/replay_client.py --modes vad300 semantic  # specific modes
    python scripts/replay_client.py --server localhost:8000   # custom server
"""

import argparse
import asyncio
import base64
import glob
import json
import os
import sys
import time
import uuid
import wave

import audioop
import websockets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CALLER_AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "caller_audio")
RECORDINGS_DIR = os.path.join(PROJECT_DIR, "data", "recordings")
SESSIONS_DIR = os.path.join(PROJECT_DIR, "data", "sessions")

ALL_MODES = ["vad300", "vad700", "semantic"]

# Timing constants
CHUNK_MS = 20  # 20ms audio chunks
ULAW_SAMPLE_RATE = 8000
ULAW_CHUNK_BYTES = int(ULAW_SAMPLE_RATE * CHUNK_MS / 1000)  # 160 bytes per 20ms
BOT_SILENCE_TIMEOUT = 3.0  # seconds of no audio = bot finished
POST_UTTERANCE_PAUSE = 0.5  # pause between utterances


# ---------------------------------------------------------------------------
# Audio conversion helpers
# ---------------------------------------------------------------------------


def wav_to_ulaw_chunks(path: str, chunk_ms: int = CHUNK_MS) -> list[bytes]:
    """Read a WAV file and return a list of 8kHz µ-law chunks.

    Each chunk represents `chunk_ms` milliseconds of audio.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        orig_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    # Convert to mono if stereo
    if n_channels == 2:
        raw = audioop.tomono(raw, sampwidth, 0.5, 0.5)

    # Convert to 16-bit if needed
    if sampwidth == 1:
        raw = audioop.lin2lin(raw, 1, 2)
        sampwidth = 2
    elif sampwidth > 2:
        raw = audioop.lin2lin(raw, sampwidth, 2)
        sampwidth = 2

    # Resample to 8kHz
    if orig_rate != ULAW_SAMPLE_RATE:
        raw, _ = audioop.ratecv(raw, sampwidth, 1, orig_rate, ULAW_SAMPLE_RATE, None)

    # Convert PCM16 to µ-law
    ulaw_data = audioop.lin2ulaw(raw, 2)

    # Split into chunks
    chunks = []
    for i in range(0, len(ulaw_data), ULAW_CHUNK_BYTES):
        chunk = ulaw_data[i : i + ULAW_CHUNK_BYTES]
        if len(chunk) == ULAW_CHUNK_BYTES:
            chunks.append(chunk)
    return chunks


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Convert µ-law bytes to 16-bit PCM at 8kHz."""
    return audioop.ulaw2lin(data, 2)


def save_wav(path: str, pcm_data: bytes, sample_rate: int = ULAW_SAMPLE_RATE, channels: int = 1):
    """Save raw PCM16 data as a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def make_stereo(left_pcm: bytes, right_pcm: bytes) -> bytes:
    """Interleave two mono PCM16 streams into stereo.

    Pads the shorter stream with silence.
    """
    # Ensure equal length (pad shorter with silence)
    max_len = max(len(left_pcm), len(right_pcm))
    left_pcm = left_pcm.ljust(max_len, b"\x00")
    right_pcm = right_pcm.ljust(max_len, b"\x00")

    # Interleave samples (each sample is 2 bytes)
    stereo = bytearray()
    for i in range(0, max_len, 2):
        stereo.extend(left_pcm[i : i + 2])
        stereo.extend(right_pcm[i : i + 2])
    return bytes(stereo)


# ---------------------------------------------------------------------------
# Plivo protocol helpers
# ---------------------------------------------------------------------------


def plivo_start_message(stream_id: str, call_id: str) -> str:
    """Create a Plivo-compatible start handshake message."""
    return json.dumps({"start": {"streamId": stream_id, "callId": call_id}})


def plivo_media_message(ulaw_chunk: bytes) -> str:
    """Create a Plivo-compatible media message with base64-encoded µ-law audio."""
    return json.dumps(
        {
            "event": "media",
            "media": {
                "payload": base64.b64encode(ulaw_chunk).decode("ascii"),
                "contentType": "audio/x-mulaw",
                "sampleRate": ULAW_SAMPLE_RATE,
            },
        }
    )


# ---------------------------------------------------------------------------
# Core replay logic
# ---------------------------------------------------------------------------


def find_latest_session(mode: str, after_ts: float) -> str | None:
    """Find the most recently modified session JSON for the given mode created after after_ts."""
    pattern = os.path.join(SESSIONS_DIR, "*.json")
    candidates = []
    for path in glob.glob(pattern):
        if os.path.getmtime(path) < after_ts:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("mode") == mode:
                candidates.append((os.path.getmtime(path), path, data))
        except (json.JSONDecodeError, KeyError):
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2].get("session_id")


def find_bot_wav(session_id: str) -> str | None:
    """Find the bot WAV file recorded by AudioRecorderProcessor."""
    path = os.path.join(RECORDINGS_DIR, f"{session_id}_bot.wav")
    return path if os.path.exists(path) else None


def read_wav_pcm(path: str) -> tuple[bytes, int, int]:
    """Read a WAV file and return (pcm_data, sample_rate, num_channels)."""
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    # Convert to 16-bit mono if needed
    if n_ch == 2:
        raw = audioop.tomono(raw, sw, 0.5, 0.5)
        n_ch = 1
    if sw != 2:
        raw = audioop.lin2lin(raw, sw, 2)
    return raw, rate, n_ch


async def run_mode(mode: str, utterance_files: list[str], server: str) -> dict:
    """Run a full conversation replay through one mode.

    Returns a summary dict with timing and recording info.
    """
    ws_url = f"ws://{server}/{mode}/ws"
    stream_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  WebSocket: {ws_url}")
    print(f"  Utterances: {len(utterance_files)}")
    print(f"{'='*60}")

    # Prepare all utterance chunks upfront
    all_utterance_chunks = []
    for path in utterance_files:
        chunks = wav_to_ulaw_chunks(path)
        all_utterance_chunks.append(chunks)
        duration_s = len(chunks) * CHUNK_MS / 1000
        print(f"  Loaded {os.path.basename(path)}: {len(chunks)} chunks ({duration_s:.1f}s)")

    # Track bot audio via WebSocket events (for timing only, not recording)
    last_audio_time = time.time()
    bot_speaking = asyncio.Event()

    # Caller timeline: list of (start_time, pcm_data) for each utterance
    caller_segments: list[tuple[float, bytes]] = []

    async def receive_loop(ws):
        """Receive messages from the WebSocket, tracking bot audio timing."""
        nonlocal last_audio_time
        try:
            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError):
                    continue

                event = msg.get("event", "")

                if event == "playAudio":
                    last_audio_time = time.time()
                    bot_speaking.set()
                elif event == "clearAudio":
                    pass  # Bot interrupted itself
        except websockets.exceptions.ConnectionClosed:
            pass

    async def wait_for_bot_silence(timeout: float = BOT_SILENCE_TIMEOUT):
        """Wait until the bot stops sending audio for `timeout` seconds."""
        # First wait for bot to start speaking
        try:
            await asyncio.wait_for(bot_speaking.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            print("    [Warning] Bot never started speaking, continuing...")
            return

        bot_speaking.clear()

        # Now wait for silence
        while True:
            elapsed = time.time() - last_audio_time
            if elapsed >= timeout:
                break
            await asyncio.sleep(0.1)

    async def send_utterance(ws, chunks: list[bytes], utterance_num: int, utterance_path: str):
        """Send one utterance's audio chunks at real-time pace."""
        duration_s = len(chunks) * CHUNK_MS / 1000
        print(f"  [{utterance_num}] Sending audio ({duration_s:.1f}s)...")
        send_start = time.time()
        for chunk in chunks:
            msg = plivo_media_message(chunk)
            await ws.send(msg)
            await asyncio.sleep(CHUNK_MS / 1000)

        # Record caller segment with its start time and 8kHz PCM data
        with wave.open(utterance_path, "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if n_ch == 2:
            raw = audioop.tomono(raw, sw, 0.5, 0.5)
        if sw != 2:
            raw = audioop.lin2lin(raw, sw, 2)
            sw = 2
        if rate != ULAW_SAMPLE_RATE:
            raw, _ = audioop.ratecv(raw, 2, 1, rate, ULAW_SAMPLE_RATE, None)
        caller_segments.append((send_start, raw))

        print(f"  [{utterance_num}] Audio sent, waiting for bot response...")

    # Record timestamp before call so we can find the session JSON created during the call
    pre_call_ts = time.time()

    # Connect and run conversation
    start_time = time.time()

    try:
        async with websockets.connect(ws_url) as ws:
            # Start receiver task
            recv_task = asyncio.create_task(receive_loop(ws))

            # Send Plivo handshake (two messages: start + dummy)
            await ws.send(plivo_start_message(stream_id, call_id))
            await ws.send(json.dumps({}))
            print("  Handshake sent, waiting for bot greeting...")

            # Wait for bot greeting to finish
            await wait_for_bot_silence()
            greeting_time = time.time() - start_time
            print(f"  Bot greeting complete ({greeting_time:.1f}s)")

            # Send each utterance
            for i, (chunks, utt_path) in enumerate(zip(all_utterance_chunks, utterance_files), 1):
                await send_utterance(ws, chunks, i, utt_path)
                await wait_for_bot_silence()
                print(f"  [{i}] Bot responded")

                if i < len(all_utterance_chunks):
                    await asyncio.sleep(POST_UTTERANCE_PAUSE)

            # Close connection
            print("  Closing connection...")
            await ws.close()
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"  Error: {e}")
        return {"mode": mode, "error": str(e)}

    elapsed = time.time() - start_time
    print(f"  Session complete ({elapsed:.1f}s total)")

    # --- Build combined recording from pipeline-captured bot audio ---
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    combined_path = os.path.join(RECORDINGS_DIR, f"{mode}_combined.wav")
    caller_path = os.path.join(RECORDINGS_DIR, f"{mode}_caller.wav")

    # Wait for pipeline cleanup to finish writing session JSON + bot WAV.
    # The CancelFrame propagates through the pipeline after WebSocket close,
    # so the files may not exist immediately.
    mode_value_map = {"vad300": "vad_only", "vad700": "vad_only", "semantic": "semantic"}
    mode_value = mode_value_map.get(mode, mode)
    session_id = None
    bot_wav_path = None
    for attempt in range(10):
        await asyncio.sleep(0.5)
        if session_id is None:
            session_id = find_latest_session(mode_value, pre_call_ts)
        if session_id and bot_wav_path is None:
            bot_wav_path = find_bot_wav(session_id)
        if session_id and bot_wav_path:
            break

    if not session_id:
        print(f"  Warning: Could not find session JSON for {mode}")
        return {"mode": mode, "elapsed_s": round(elapsed, 1), "error": "No session JSON found"}

    if not bot_wav_path:
        print(f"  Warning: No bot WAV found for session {session_id}")
        return {"mode": mode, "elapsed_s": round(elapsed, 1), "error": "No bot WAV found"}

    print(f"  Found bot audio: {bot_wav_path}")

    # Read bot audio (may be higher sample rate than 8kHz)
    bot_pcm, bot_rate, _ = read_wav_pcm(bot_wav_path)

    # Resample bot audio to 8kHz to match caller
    if bot_rate != ULAW_SAMPLE_RATE:
        bot_pcm, _ = audioop.ratecv(bot_pcm, 2, 1, bot_rate, ULAW_SAMPLE_RATE, None)
        print(f"  Resampled bot audio from {bot_rate}Hz to {ULAW_SAMPLE_RATE}Hz")

    # Build caller timeline (8kHz mono PCM)
    if caller_segments:
        t0 = caller_segments[0][0]
        # Calculate total duration from first caller segment to end of last
        t_end = t0
        for seg_t, seg_data in caller_segments:
            seg_dur = len(seg_data) / (ULAW_SAMPLE_RATE * 2)
            t_end = max(t_end, seg_t + seg_dur)

        # Add enough room for bot audio too
        bot_dur = len(bot_pcm) / (ULAW_SAMPLE_RATE * 2)
        total_dur = max(t_end - t0, bot_dur) + 0.5
        total_samples = int(total_dur * ULAW_SAMPLE_RATE)
        bytes_per_sample = 2

        caller_pcm = bytearray(total_samples * bytes_per_sample)
        for seg_t, seg_data in caller_segments:
            offset = int((seg_t - t0) * ULAW_SAMPLE_RATE) * bytes_per_sample
            end = min(offset + len(seg_data), len(caller_pcm))
            caller_pcm[offset:end] = seg_data[: end - offset]
        caller_pcm = bytes(caller_pcm)
    else:
        caller_pcm = b"\x00" * len(bot_pcm)

    # Save caller track
    save_wav(caller_path, caller_pcm)
    caller_duration = len(caller_pcm) / (ULAW_SAMPLE_RATE * 2)
    print(f"  Saved {caller_path} ({caller_duration:.1f}s)")

    # Stereo combined: caller=left, bot=right
    stereo = make_stereo(caller_pcm, bot_pcm)
    save_wav(combined_path, stereo, channels=2)
    print(f"  Saved {combined_path} (stereo, {max(caller_duration, len(bot_pcm)/(ULAW_SAMPLE_RATE*2)):.1f}s)")

    bot_duration = len(bot_pcm) / (ULAW_SAMPLE_RATE * 2)
    return {
        "mode": mode,
        "elapsed_s": round(elapsed, 1),
        "bot_audio_s": round(bot_duration, 1),
        "utterances": len(utterance_files),
        "caller_path": caller_path,
        "combined_path": combined_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_utterance_files() -> list[str]:
    """Find utterance WAV files in data/caller_audio/, sorted by name."""
    if not os.path.isdir(CALLER_AUDIO_DIR):
        print(f"Error: Caller audio directory not found: {CALLER_AUDIO_DIR}")
        print("Run: python scripts/generate_caller_audio.py")
        sys.exit(1)

    files = sorted(
        os.path.join(CALLER_AUDIO_DIR, f)
        for f in os.listdir(CALLER_AUDIO_DIR)
        if f.endswith(".wav") and f.startswith("utterance_")
    )

    if not files:
        print(f"Error: No utterance_*.wav files in {CALLER_AUDIO_DIR}")
        print("Run: python scripts/generate_caller_audio.py")
        sys.exit(1)

    return files


async def async_main(modes: list[str], server: str, runs: int = 1):
    utterance_files = find_utterance_files()
    print(f"Found {len(utterance_files)} utterance files")
    print(f"Server: {server}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Runs per mode: {runs}")
    total_calls = len(modes) * runs
    print(f"Total calls: {total_calls}")

    results = []
    call_num = 0
    for mode in modes:
        for run in range(1, runs + 1):
            call_num += 1
            print(f"\n[Call {call_num}/{total_calls}] {mode} run {run}/{runs}")
            result = await run_mode(mode, utterance_files, server)
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for mode in modes:
        mode_results = [r for r in results if r.get("mode") == mode]
        successes = [r for r in mode_results if "error" not in r]
        errors = [r for r in mode_results if "error" in r]
        if successes:
            avg_time = sum(r["elapsed_s"] for r in successes) / len(successes)
            print(f"  {mode}: {len(successes)}/{len(mode_results)} OK, avg {avg_time:.1f}s per call")
        if errors:
            print(f"  {mode}: {len(errors)} errors")
    print()


def main():
    parser = argparse.ArgumentParser(description="Replay caller audio through turn detection modes")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ALL_MODES,
        default=ALL_MODES,
        help="Modes to test (default: all)",
    )
    parser.add_argument(
        "--server",
        default="localhost:8000",
        help="multi_bot.py server address (default: localhost:8000)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of calls per mode (default: 1)",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args.modes, args.server, args.runs))


if __name__ == "__main__":
    main()
