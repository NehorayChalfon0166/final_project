#!/usr/bin/env bash
# ============================================================
# Bitcoin Wallet Risk Analyzer — macOS/Linux One-Click Installer
# ============================================================
# First run: Creates venv and installs dependencies (~2 min)
# Subsequent runs: Skips install and runs directly
#
# Usage:
#   ./setup_and_run.sh
#   ./setup_and_run.sh 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
#   ./setup_and_run.sh --no-charts bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
# ============================================================

set -e

VENV_DIR=".analyzer_env"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Find Python 3
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "  ERROR: Python 3.10+ is required but not found."
    echo ""
    echo "  Install Python:"
    echo "    macOS:  brew install python@3.12"
    echo "    Ubuntu: sudo apt install python3.12 python3.12-venv"
    echo "    Fedora: sudo dnf install python3.12"
    echo ""
    exit 1
fi

echo "  Python $($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+') detected."

# Create venv if it doesn't exist
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo ""
    echo "  First run — setting up environment..."
    echo "  [1/3] Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"

    echo "  [2/3] Upgrading pip..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet

    echo "  [3/3] Installing dependencies (this may take a few minutes)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    pip install torch-geometric --quiet
    pip install requests matplotlib numpy --quiet

    echo ""
    echo "  Setup complete!"
    echo ""
else
    source "$VENV_DIR/bin/activate"
fi

# Run the analyzer, passing through all arguments
python wallet_analyzer.py "$@"
