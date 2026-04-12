#!/usr/bin/env bash
# Manually promote a specific model version to 'current'.
# Usage: ./scripts/deploy_model.sh v_20260301_120000
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <version>"
    echo "Available versions:"
    ls -1d "${TC_MODEL_STORE_PATH:-models}"/v_* 2>/dev/null | xargs -I{} basename {} || echo "  (none)"
    exit 1
fi

VERSION="$1"
STORE="${TC_MODEL_STORE_PATH:-models}"

if [ ! -d "$STORE/$VERSION" ]; then
    echo "Error: Version directory not found: $STORE/$VERSION"
    exit 1
fi

# Atomic symlink swap
ln -sfn "$(cd "$STORE/$VERSION" && pwd)" "$STORE/.current_tmp_$$"
mv -f "$STORE/.current_tmp_$$" "$STORE/current"

echo "Promoted $VERSION to current"
echo "The API will pick up the new model within 60 seconds."
