#!/bin/bash
# One-click launcher for macOS. Download just THIS file and double-click it.
# It downloads the analyzer script (if missing), builds an isolated Python
# environment, and runs the analyzer inside it.
set -e

cd "$(dirname "$0")"

SCRIPT="wallet_analyzer.py"
SCRIPT_URL="https://raw.githubusercontent.com/NehorayChalfon0166/final_project/chore/cleanup-and-consolidate/standalone/wallet_analyzer.py"
PY="${PYTHON:-python3}"
VENV=".venv"

# --- 1. Download the analyzer script if it isn't next to this launcher ---
if [ ! -f "$SCRIPT" ]; then
    echo "Downloading $SCRIPT ..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$SCRIPT" "$SCRIPT_URL"
    else
        "$PY" -c "import urllib.request; urllib.request.urlretrieve('$SCRIPT_URL', '$SCRIPT')"
    fi
fi

# --- 2. Create an isolated environment the first time ---
if [ ! -d "$VENV" ]; then
    echo "Creating isolated environment (first run only) ..."
    "$PY" -m venv "$VENV"
fi

# --- 3. Run it (the script installs any missing libraries into this venv) ---
"$VENV/bin/python" "$SCRIPT" "$@"

echo ""
read -n 1 -s -r -p "Press any key to close..."
echo ""
