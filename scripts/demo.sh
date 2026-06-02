#!/usr/bin/env bash
#
# Demo: train an XGBoost model, promote it atomically (models/current symlink), serve it,
# and get ranked top-K account-code predictions with SHAP explanations.
# Records cleanly with asciinema or vhs.
#
# Prereqs: `uv`, `curl`, and `jq` on PATH. Training takes ~30s.
# Usage:   ./scripts/demo.sh            (set DEMO_PAUSE=0 to remove the pacing pauses)
#
# Note: with no API keys configured, the server runs open (it logs a startup warning) so
# no X-API-Key header is needed for this local demo.
#
set -euo pipefail
cd "$(dirname "$0")/.."

API="http://localhost:8000"
PAUSE="${DEMO_PAUSE:-1.5}"

command -v jq >/dev/null || { echo "this demo needs 'jq' (e.g. brew install jq)"; exit 1; }

echo "▶ Installing deps (uv sync --extra all)…"
uv sync --extra all >/dev/null

echo "▶ Generating synthetic transactions…"
uv run python scripts/generate_sample_data.py

echo
echo "▶ Training XGBoost and promoting it atomically…"
uv run tc-train --auto-promote -v
echo "  models/current → $(readlink models/current 2>/dev/null || echo '(none)')"
sleep "$PAUSE"

echo
echo "▶ Starting the inference API…"
uv run tc-serve >/tmp/tc-demo-serve.log 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "$API/ready" >/dev/null 2>&1 && break
  sleep 0.5
done
sleep "$PAUSE"

echo
echo "▶ Classify a transaction — ranked top-3 account codes with confidence:"
curl -s -X POST "$API/classify" -H "Content-Type: application/json" -d '{
  "transactions": [
    {"description":"URSSAF COTISATIONS","remarks":"PRLV SEPA CPY:FR123","debit":1234.56,"posting_date":"2025-01-15"}
  ],
  "top_k": 3
}' | jq
sleep "$PAUSE"

echo
echo "▶ Explain the prediction — top SHAP feature contributions:"
curl -s -X POST "$API/explain?max_features=5" -H "Content-Type: application/json" -d '{
  "transactions": [
    {"description":"URSSAF COTISATIONS","remarks":"PRLV SEPA CPY:FR123","debit":1234.56,"posting_date":"2025-01-15"}
  ]
}' | jq

echo
echo "Done. (API stopped on exit)"
