#!/usr/bin/env bash
PROXY="${1:?Usage: $0 <ngrok-hostname> [port]}"
PORT="${2:-8000}"
SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())") \
  python multi_bot.py --proxy "$PROXY" --port "$PORT"
