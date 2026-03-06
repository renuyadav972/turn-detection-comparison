#!/usr/bin/env python3
"""
Generate synthetic session data for dashboard testing.

Creates 60 session JSON files: 20 each for VAD 300ms, VAD 700ms, and Semantic.
Each session uses a randomly selected personal assistant conversation script.

Usage:
    python scripts/generate_test_data.py
"""

import glob
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sessions")

# ─── Conversation Scripts (Personal Assistant) ──────────────────────

CONVERSATIONS = [
    # Script 1: Weather + reminders + calendar
    [
        ("Yeah hi, can you tell me what the weather is like in San Francisco this weekend?", "It looks like San Francisco will be partly cloudy on Saturday with a high of sixty-two, and sunny on Sunday reaching sixty-seven degrees."),
        ("Hmm okay... should I bring a jacket?", "I'd recommend a light jacket for the evenings. Temperatures will drop to the low fifties after sunset."),
        ("Oh right, and um... I need to set a reminder to call my dentist on Monday morning.", "Done! I've set a reminder for Monday at nine AM to call your dentist."),
        ("What do I have on my calendar for Monday?", "On Monday you have a team standup at ten, lunch with Sarah at noon, and now your dentist call reminder at nine AM."),
        ("Can you move the standup to ten thirty?", "Sure, I've moved your team standup to ten thirty AM on Monday."),
        ("Perfect, that's all. Thanks!", "You're welcome! Have a great weekend."),
    ],
    # Script 2: Travel planning + unit conversions
    [
        ("Yeah so I'm thinking about a trip to Tokyo next month. What's the best time to visit?", "March and April are wonderful for Tokyo — cherry blossom season starts in late March and the weather is mild, around fifty-five to sixty-five degrees."),
        ("How long is the flight from San Francisco?", "A direct flight from San Francisco to Tokyo is about eleven hours. With a layover, it can be thirteen to fifteen hours."),
        ("Actually wait, could you help me convert fifty euros to US dollars?", "Fifty euros is approximately fifty-four US dollars at current exchange rates."),
        ("And how about... um... two hundred thousand yen?", "Two hundred thousand Japanese yen is approximately one thousand three hundred and thirty US dollars."),
        ("What's a good budget for a week in Tokyo?", "For a comfortable trip, I'd budget around two thousand to three thousand dollars for a week, including hotels, food, and transportation."),
        ("Great, that's really helpful. Thanks!", "Happy to help! Let me know when you're ready to book flights."),
    ],
    # Script 3: Smart home + shopping lists
    [
        ("Hi, can you check what the temperature in the house is right now?", "Your thermostat reads seventy-two degrees. The upstairs is slightly warmer at seventy-four."),
        ("Can you turn it down to sixty-eight?", "Done, I've set the thermostat to sixty-eight degrees. It should reach that temperature in about fifteen minutes."),
        ("Add milk and eggs to my shopping list.", "I've added milk and eggs to your shopping list. You now have six items on the list."),
        ("What else is on there?", "Your shopping list has: bread, chicken breast, olive oil, spinach, milk, and eggs."),
        ("Oh wait, um... actually remove the spinach, I still have some.", "Got it, I've removed spinach from your shopping list. You're down to five items."),
        ("Can you also turn on the porch lights?", "The porch lights are now on. Would you like me to set them on a timer?"),
        ("Yeah, turn them off at eleven PM.", "Done! The porch lights will automatically turn off at eleven PM tonight."),
    ],
    # Script 4: Mixed tasks — quick-fire questions
    [
        ("Hey yeah, what time is it in London right now?", "It's currently eight fifteen PM in London."),
        ("Set a timer for twenty minutes.", "Your twenty-minute timer is set and counting down."),
        ("How do you spell... um... the word 'Mediterranean'?", "Mediterranean is spelled M-E-D-I-T-E-R-R-A-N-E-A-N."),
        ("What's the square root of one forty-four?", "The square root of one hundred forty-four is twelve."),
        ("Can you read me today's top news headline?", "The top headline today is about the Federal Reserve holding interest rates steady at their latest meeting."),
        ("Interesting. That's all I needed. Thanks for the help!", "You're welcome! Have a great rest of your day."),
    ],
]

REPROMPT_RESPONSES = [
    "I'm sorry, could you repeat that?",
    "Sorry, I didn't catch that. Could you say that again?",
    "I didn't get that, could you repeat?",
]

# Interrupted bot text — bot gets cut off mid-sentence
INTERRUPTED_FRAGMENTS = [
    "Sure, let me—",
    "I can help with—",
    "The answer to that—",
    "Okay so—",
    "Right, so what you—",
    "Let me check—",
    "Yes, the—",
]

# Mode-dependent probabilities: (interruption, false_endpoint, reprompt)
FLAG_PROBABILITIES = {
    ("vad_only", 0.3): (0.25, 0.35, 0.12),
    ("vad_only", 0.7): (0.08, 0.12, 0.05),
    ("smart_turn", None): (0.02, 0.05, 0.02),
    ("flux_semantic", None): (0.04, 0.06, 0.03),
    ("semantic", None): (0.02, 0.05, 0.02),  # legacy
}


def generate_session(mode: str, vad_stop_secs: float, base_time: float):
    session_id = str(uuid.uuid4())
    # Pick a random conversation and random turn count (4-8)
    script = random.choice(CONVERSATIONS)
    num_turns = random.randint(4, min(8, len(script)))
    conversation = script[:num_turns]

    turns = []
    flag_key = (mode, None) if mode in ("semantic", "smart_turn", "flux_semantic") else (mode, vad_stop_secs)
    int_prob, fe_prob, rp_prob = FLAG_PROBABILITIES.get(flag_key, (0.05, 0.10, 0.03))

    cursor = base_time  # tracks the timeline; next event starts here

    for i, (user_text, bot_text) in enumerate(conversation):
        turn_num = i + 1

        # Per-turn flags (decide before timestamps so we can adjust timing)
        is_interruption = (i > 0) and (random.random() < int_prob)
        is_false_endpoint = (i > 0) and (random.random() < fe_prob)
        is_reprompt = random.random() < rp_prob

        # Don't stack interruption + false endpoint on same turn
        if is_interruption and is_false_endpoint:
            is_false_endpoint = False

        # User starts speaking after bot finishes + a natural pause
        if is_interruption and turns:
            # Interruption: bot starts responding too early, user wasn't done.
            # The bot spoke briefly before user continued — no timeline overlap,
            # but the bot's short fragment counts as a wasted turn.
            prev_bot_stopped = turns[-1]["bot_stopped_at"]
            user_started = prev_bot_stopped + random.uniform(0.8, 2.0)
        elif is_false_endpoint and turns:
            # False endpoint: system thought user was done during a pause,
            # but user continues after a beat. Bot jumped in too early.
            prev_bot_stopped = turns[-1]["bot_stopped_at"]
            user_started = prev_bot_stopped + random.uniform(0.5, 1.5)
        else:
            # Normal: user waits for bot to finish, then pauses before speaking
            user_started = cursor + random.uniform(0.8, 2.5)

        speak_duration = len(user_text) * 0.04 + random.uniform(0.3, 0.8)
        user_stopped = user_started + speak_duration

        # Response latency modeling (simplified — no pipeline breakdown)
        if mode in ("semantic", "smart_turn"):
            # Smart Turn: 300-600ms, adaptive to utterance clarity
            base_latency = random.uniform(300, 600)
            if "..." in user_text or "um" in user_text.lower():
                base_latency += random.uniform(50, 150)  # hesitation → slightly slower
        elif mode == "flux_semantic":
            # Flux semantic: 200-450ms, cloud-based turn detection
            base_latency = random.uniform(200, 450)
            if "..." in user_text or "um" in user_text.lower():
                base_latency += random.uniform(30, 100)
        else:
            # VAD: threshold + 150-350ms processing
            processing = random.uniform(150, 350)
            base_latency = vad_stop_secs * 1000 + processing

        latency_ms = round(max(base_latency, 150))
        bot_started = user_stopped + latency_ms / 1000

        # Determine bot text
        if is_reprompt:
            actual_bot_text = random.choice(REPROMPT_RESPONSES)
        elif is_interruption:
            # Bot got cut off — only a fragment was spoken
            actual_bot_text = random.choice(INTERRUPTED_FRAGMENTS)
        else:
            actual_bot_text = bot_text

        bot_speak_duration = len(actual_bot_text) * 0.04 + random.uniform(0.3, 0.8)
        bot_stopped = bot_started + bot_speak_duration

        # Advance cursor to when the bot finishes
        cursor = bot_stopped

        turn = {
            "turn_number": turn_num,
            "user_started_at": round(user_started, 3),
            "user_stopped_at": round(user_stopped, 3),
            "bot_started_at": round(bot_started, 3),
            "bot_stopped_at": round(bot_stopped, 3),
            "response_latency_ms": latency_ms,
            "user_text": user_text,
            "bot_text": actual_bot_text,
            "is_interruption": is_interruption,
            "is_false_endpoint": is_false_endpoint,
            "is_reprompt": is_reprompt,
        }

        turns.append(turn)

    # Build summary
    latencies = [t["response_latency_ms"] for t in turns]
    num_t = len(turns)
    denom_pairs = max(num_t - 1, 1)

    interruption_count = sum(1 for t in turns if t["is_interruption"])
    false_endpoint_count = sum(1 for t in turns if t["is_false_endpoint"])
    reprompt_count = sum(1 for t in turns if t["is_reprompt"])
    dead_air_ms = sum(latencies)

    summary = {
        "total_turns": num_t,
        "avg_response_latency_ms": round(sum(latencies) / len(latencies)),
        "interruption_count": interruption_count,
        "interruption_rate": round(interruption_count / denom_pairs, 4),
        "false_endpoint_count": false_endpoint_count,
        "false_endpoint_rate": round(false_endpoint_count / denom_pairs, 4),
        "reprompt_count": reprompt_count,
        "reprompt_rate": round(reprompt_count / max(num_t, 1), 4),
        "dead_air_ms": dead_air_ms,
        "dead_air_s": round(dead_air_ms / 1000, 2),
    }

    started_at = datetime.fromtimestamp(base_time, tz=timezone.utc).isoformat()
    # Estimate call duration from turn timestamps
    first_user = turns[0]["user_started_at"] if turns else base_time
    last_bot = turns[-1]["bot_stopped_at"] if turns else base_time + 70
    call_secs = round(last_bot - first_user, 2)
    ended_at = datetime.fromtimestamp(base_time + call_secs + 5, tz=timezone.utc).isoformat()

    session = {
        "session_id": session_id,
        "mode": mode,
        "started_at": started_at,
        "ended_at": ended_at,
        "config": {
            "vad_stop_secs": vad_stop_secs,
            "smart_turn_stop_secs": 3.0 if mode in ("semantic", "smart_turn") else None,
        },
        "turns": turns,
        "summary": summary,
    }

    return session


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Clear existing sessions
    existing = glob.glob(os.path.join(DATA_DIR, "*.json"))
    if existing:
        print(f"Clearing {len(existing)} existing session files...")
        for f in existing:
            os.remove(f)

    # 20 sessions each for 4 modes
    configs = [
        ("vad_only", 0.3, "VAD 300ms"),
        ("vad_only", 0.7, "VAD 700ms"),
        ("smart_turn", 0.7, "Smart Turn"),
        ("flux_semantic", 0.7, "Flux Semantic"),
    ]

    sessions_per_mode = 20
    now = time.time()
    total = 0

    for mode, vad_stop, label in configs:
        print(f"\nGenerating {sessions_per_mode} sessions for {label}:")
        for i in range(sessions_per_mode):
            # Stagger timestamps: each session ~5 minutes apart
            base_time = now - (sessions_per_mode - i) * 300
            session = generate_session(mode, vad_stop, base_time)
            path = os.path.join(DATA_DIR, f"{session['session_id']}.json")
            with open(path, "w") as f:
                json.dump(session, f, indent=2)
            avg = session["summary"]["avg_response_latency_ms"]
            turns = session["summary"]["total_turns"]
            print(f"  [{i+1:2d}] {turns} turns, avg={avg}ms  {session['session_id'][:8]}")
            total += 1

    print(f"\n{total} sessions written to {DATA_DIR}")


if __name__ == "__main__":
    main()
