#!/usr/bin/env bash
# Manually promote a specific model version to 'current'.
# Usage: ./scripts/deploy_model.sh v-20260301-120000
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <version>"
    echo "Available versions:"
    ls -1d "${TXCLS_ARTIFACT_DIR:-models}"/v-* 2>/dev/null | xargs -I{} basename {} || echo "  (none)"
    exit 1
fi

VERSION="$1"
STORE="${TXCLS_ARTIFACT_DIR:-models}"

if [ ! -d "$STORE/$VERSION" ]; then
    echo "Error: Version directory not found: $STORE/$VERSION"
    exit 1
fi

# Atomic symlink swap
ln -sfn "$(cd "$STORE/$VERSION" && pwd)" "$STORE/.current_tmp_$$"
mv -f "$STORE/.current_tmp_$$" "$STORE/current"

echo "Promoted $VERSION to current"
echo "The API will pick up the new model within a few seconds (watchdog + debounce)."
