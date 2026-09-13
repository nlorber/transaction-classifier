#!/usr/bin/env bash
# Manually promote a specific model version to 'current'.
# Usage: ./scripts/deploy_model.sh v-20260301-120000
#
# Delegates to ModelStore.promote, the same atomic symlink swap that
# `tc-train --auto-promote` uses. TXCLS_PYTHON overrides the interpreter
# (default: uv run python).
set -euo pipefail

STORE="${TXCLS_ARTIFACT_DIR:-models}"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <version>"
    echo "Available versions:"
    ls -1d "$STORE"/v-* 2>/dev/null | xargs -I{} basename {} || echo "  (none)"
    exit 1
fi

VERSION="$1"

if [ ! -d "$STORE/$VERSION" ]; then
    echo "Error: Version directory not found: $STORE/$VERSION"
    exit 1
fi

# Unquoted on purpose: the default is a multi-word command.
${TXCLS_PYTHON:-uv run python} -c '
import sys
from transaction_classifier.core.artifacts.store import ModelStore
ModelStore(sys.argv[1]).promote(sys.argv[2])
' "$STORE" "$VERSION"

echo "Promoted $VERSION to current"
echo "The API will pick up the new model within a few seconds (watchdog + debounce)."
