#!/usr/bin/env bash
# Trigger an outbound call via the Pipecat runner's /start endpoint.
#
# Usage:
#   ./scripts/test_call.sh +14155551234
#
# The runner exposes POST /start which initiates a new pipeline session.
# For inbound calls, point your Plivo number's Answer URL to
# https://<ngrok-host>/answer instead.

set -euo pipefail

PHONE="${1:?Usage: $0 <phone-number>}"
HOST="${2:-http://localhost:8000}"

curl -s -X POST "${HOST}/start" \
  -H "Content-Type: application/json" \
  -d "{\"to\": \"${PHONE}\"}" | python3 -m json.tool
