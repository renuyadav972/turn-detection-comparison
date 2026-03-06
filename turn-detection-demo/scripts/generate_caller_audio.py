"""
Generate Caller Audio
=====================
Uses ElevenLabs TTS to generate individual caller utterance WAV files
for replay testing. Uses a different voice from the bot (Adam instead of Rachel).

Usage:
    python scripts/generate_caller_audio.py
"""

import os
import sys
import wave

import requests
from dotenv import load_dotenv

load_dotenv(override=False)

# Adam voice (male) — distinct from bot's Rachel voice
CALLER_VOICE_ID = "pNInz6obpgDQGcFmaJgB"

UTTERANCES = [
    "Yeah hi, can you tell me what the weather is like in San Francisco this weekend?",
    "Hmm okay... can you also remind me, what time does the Golden Gate Bridge visitor center close?",
    "Oh right, and um... I need to set a reminder to call my dentist on Monday morning.",
    "Actually wait, could you help me convert fifty euros to US dollars?",
    "Great, that's all I needed. Thanks for the help!",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "caller_audio")


PCM_SAMPLE_RATE = 22050


def generate_utterance(text: str, output_path: str, api_key: str):
    """Generate a single utterance WAV file via ElevenLabs API."""
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{CALLER_VOICE_ID}"
        f"?output_format=pcm_22050"
    )
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()

    # ElevenLabs pcm_22050 returns raw 16-bit PCM — wrap in WAV header
    pcm_data = resp.content
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(PCM_SAMPLE_RATE)
        wf.writeframes(pcm_data)

    size_kb = os.path.getsize(output_path) / 1024
    duration_s = len(pcm_data) / (PCM_SAMPLE_RATE * 2)
    print(f"  Saved {output_path} ({size_kb:.1f} KB, {duration_s:.1f}s)")


def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not set in .env")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating {len(UTTERANCES)} caller utterances...")
    print(f"Voice: Adam ({CALLER_VOICE_ID})")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}\n")

    for i, text in enumerate(UTTERANCES, 1):
        filename = f"utterance_{i:02d}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        print(f"[{i}/{len(UTTERANCES)}] \"{text}\"")
        generate_utterance(text, output_path, api_key)

    print(f"\nDone! {len(UTTERANCES)} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
