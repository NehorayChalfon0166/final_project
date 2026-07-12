#!/bin/bash
# One-click launcher for macOS. Double-click this file in Finder.
# It creates an isolated Python environment the first time, then runs the
# wallet analyzer inside it (so nothing touches your system Python).
set -e

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating isolated environment (first run only) ..."
    "$PY" -m venv "$VENV"
fi

# The script itself installs any missing libraries into this venv.
"$VENV/bin/python" wallet_analyzer.py "$@"

echo ""
read -n 1 -s -r -p "Press any key to close..."
echo ""
