#!/bin/bash
# Q23: Double-click runner (diagnostics -> strategies -> dashboard) for macOS

set -euo pipefail

# Resolve project directory (script location wins)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${Q23_PROJECT_DIR:-$SCRIPT_DIR}"

echo
echo "=== Q23 Runner ==="
echo "Project: $PROJECT_DIR"
echo

# ------------------------------------------------------------------
# Project dir sanity
# ------------------------------------------------------------------
if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project directory not found: $PROJECT_DIR"
  echo
  read -n 1 -s -r -p "Press any key to close..."
  echo
  exit 1
fi
cd "$PROJECT_DIR"

# ------------------------------------------------------------------
# Load .env (KEY=VALUE, no export required)
# ------------------------------------------------------------------
if [ -f "$PROJECT_DIR/.env" ]; then
  echo "Loading .env"
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_DIR/.env"
  set +a
else
  echo "WARNING: .env not found at $PROJECT_DIR/.env"
  echo "You likely need API_KEY in .env for Quantiacs."
  echo
fi

# ------------------------------------------------------------------
# Conda config (allow .env overrides)
# ------------------------------------------------------------------
CONDA_SH="${Q23_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${Q23_CONDA_ENV:-qntdev}"

if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: conda.sh not found at:"
  echo "  $CONDA_SH"
  echo "Fix Q23_CONDA_SH in .env."
  echo
  read -n 1 -s -r -p "Press any key to close..."
  echo
  exit 1
fi

# Load conda
# shellcheck disable=SC1090
source "$CONDA_SH"

# Activate env
conda activate "$CONDA_ENV"

# ------------------------------------------------------------------
# Ensure imports work
# ------------------------------------------------------------------
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

# Strategy toggles (defaults)
export Q23_RUN_V4="${Q23_RUN_V4:-true}"
export Q23_RUN_NASNYS1010="${Q23_RUN_NASNYS1010:-false}"
export Q23_RUN_Q23LSNEW3="${Q23_RUN_Q23LSNEW3:-false}"

# ------------------------------------------------------------------
# Diagnostics (FAIL FAST)
# ------------------------------------------------------------------
echo
echo "===== Environment diagnostics ====="

if [ ! -f "$PROJECT_DIR/diagnose_env.py" ]; then
  echo "ERROR: diagnose_env.py not found in project root."
  echo "This file is required for startup diagnostics."
  echo
  read -n 1 -s -r -p "Press any key to close..."
  echo
  exit 1
fi

python diagnose_env.py || {
  echo
  echo "ERROR: Environment diagnostics failed."
  echo "Fix the issues above before running strategies."
  echo
  read -n 1 -s -r -p "Press any key to close..."
  echo
  exit 1
}

# ------------------------------------------------------------------
# Run strategies
# ------------------------------------------------------------------
echo
echo "===== Running strategies ====="
python -u run_strategy.py
echo "===== Strategy runs complete ====="
echo

# ------------------------------------------------------------------
# Launch dashboard
# ------------------------------------------------------------------
echo "===== Launching dashboard ====="
python -u run_dashboard.py || true

echo
echo "=== All tasks finished ==="
read -n 1 -s -r -p "Press any key to close..."
echo
