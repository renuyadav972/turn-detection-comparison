#!/usr/bin/env python3
"""
Generate synthetic session data for dashboard testing.

Creates 5 session JSON files matching the test matrix:
  1. VAD 300ms  — aggressive, fast but high false-trigger
  2. VAD 500ms  — moderate
  3. VAD 700ms  — conservative default
  4. VAD 1000ms — very safe but sluggish
  5. Semantic   — Smart Turn v3, context-aware

Usage:
    python scripts/generate_test_data.py
"""

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sessions")

# Simulated conversation turns (same script across all configs)
CONVERSATION = [
    ("Hi, I'd like to order a pizza please.", "Hi there! What kind of pizza would you like to order today?"),
    ("Um... I think I'll have a large pepperoni.", "Great choice! A large pepperoni pizza. Would you like any extra toppings on that?"),
    ("Actually, could you... could you add mushrooms and green peppers?", "Sure thing! I'll add mushrooms and green peppers to your large pepperoni. Anything else?"),
    ("Yeah, and also a side of garlic bread.", "Perfect, a side of garlic bread. Would you like any drinks with that?"),
    ("No thanks, that's everything. How much will that be?", "Your total comes to eighteen dollars and fifty cents. Would you like to pay by card?"),
]


def generate_session(mode: str, vad_stop_secs: float, smart_turn_stop_secs: float = 3.0):
    session_id = str(uuid.uuid4())
    base_time = time.time() - 300  # 5 minutes ago
    turns = []

    for i, (user_text, bot_text) in enumerate(CONVERSATION):
        turn_num = i + 1
        user_started = base_time + i * 12
        # Simulate speaking duration proportional to text length
        speak_duration = len(user_text) * 0.04 + random.uniform(0.3, 0.8)
        user_stopped = user_started + speak_duration

        # Response latency depends on mode and threshold
        if mode == "semantic":
            # Semantic: generally fast, adaptive to context
            base_latency = random.uniform(380, 620)
            # Hesitation turns get lower latency (model waits appropriately)
            if "..." in user_text or "could you" in user_text:
                base_latency = random.uniform(550, 750)
        else:
            # VAD: latency = threshold + processing overhead
            # Lower threshold = faster but more false triggers
            processing_overhead = random.uniform(200, 350)
            base_latency = vad_stop_secs * 1000 + processing_overhead
            # Add jitter
            base_latency += random.uniform(-50, 80)

        latency_ms = round(max(base_latency, 150))
        bot_started = user_stopped + latency_ms / 1000

        turn = {
            "turn_number": turn_num,
            "user_started_at": round(user_started, 3),
            "user_stopped_at": round(user_stopped, 3),
            "bot_started_at": round(bot_started, 3),
            "response_latency_ms": latency_ms,
            "user_text": user_text,
            "bot_text": bot_text,
        }

        if mode == "semantic":
            # Simulate Smart Turn confidence
            # Higher confidence for clear complete sentences
            if user_text.endswith("?") or user_text.endswith("."):
                prob = random.uniform(0.85, 0.98)
            elif "..." in user_text:
                prob = random.uniform(0.55, 0.75)
            else:
                prob = random.uniform(0.70, 0.90)

            turn["smart_turn"] = {
                "is_complete": prob > 0.5,
                "probability": round(prob, 4),
                "inference_time_ms": round(random.uniform(45, 85), 1),
                "e2e_processing_time_ms": round(random.uniform(65, 110), 1),
            }

        turns.append(turn)

    # Build summary
    latencies = [t["response_latency_ms"] for t in turns]
    summary = {
        "total_turns": len(turns),
        "avg_response_latency_ms": round(sum(latencies) / len(latencies)),
        "min_response_latency_ms": min(latencies),
        "max_response_latency_ms": max(latencies),
    }

    smart_probs = [t["smart_turn"]["probability"] for t in turns if "smart_turn" in t]
    smart_infer = [t["smart_turn"]["inference_time_ms"] for t in turns if "smart_turn" in t]
    if smart_probs:
        summary["avg_smart_turn_probability"] = round(sum(smart_probs) / len(smart_probs), 4)
    if smart_infer:
        summary["avg_smart_turn_inference_ms"] = round(sum(smart_infer) / len(smart_infer), 1)

    started_at = datetime.fromtimestamp(base_time, tz=timezone.utc).isoformat()
    ended_at = datetime.fromtimestamp(base_time + 70, tz=timezone.utc).isoformat()

    session = {
        "session_id": session_id,
        "mode": mode,
        "started_at": started_at,
        "ended_at": ended_at,
        "config": {
            "vad_stop_secs": vad_stop_secs,
            "smart_turn_stop_secs": smart_turn_stop_secs,
        },
        "turns": turns,
        "summary": summary,
    }

    return session


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    configs = [
        ("vad_only", 0.3),
        ("vad_only", 0.5),
        ("vad_only", 0.7),
        ("vad_only", 1.0),
        ("semantic", 0.7),
    ]

    for mode, vad_stop in configs:
        session = generate_session(mode, vad_stop)
        path = os.path.join(DATA_DIR, f"{session['session_id']}.json")
        with open(path, "w") as f:
            json.dump(session, f, indent=2)
        label = f"Semantic" if mode == "semantic" else f"VAD {int(vad_stop * 1000)}ms"
        print(f"  {label:20s} -> {session['session_id']}  avg={session['summary']['avg_response_latency_ms']}ms")

    print(f"\n{len(configs)} sessions written to {DATA_DIR}")


if __name__ == "__main__":
    main()
