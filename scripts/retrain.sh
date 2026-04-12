#!/usr/bin/env bash
# Monthly retraining cron wrapper.
# Usage: Add to crontab:
#   0 2 1 * * /path/to/transaction-classifier/scripts/retrain.sh >> /var/log/tc-retrain.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "$(date -Iseconds) Starting retraining..."

# Run training with auto-promote (validates before promoting)
uv run tc-train --auto-promote --verbose

echo "$(date -Iseconds) Retraining complete."
