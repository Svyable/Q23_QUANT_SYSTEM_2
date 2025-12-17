#!/bin/bash
# Q23: Run strategies then launch dashboard (double-click friendly on macOS)
set -euo pipefail

PROJECT_DIR="${Q23_PROJECT_DIR:-/Users/svenbenson/Q23_Quant_System}"

ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "WARNING: .env not found at $ENV_FILE"
fi

CONDA_SH="${Q23_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${Q23_CONDA_ENV:-qntdev}"

# Strategy toggles (true/false). Default: v4 on, legacy off.
export Q23_RUN_V4="${Q23_RUN_V4:-true}"
export Q23_RUN_NASNYS1010="${Q23_RUN_NASNYS1010:-false}"
export Q23_RUN_Q23LSNEW3="${Q23_RUN_Q23LSNEW3:-false}"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
else
  echo "Could not find conda init script at $CONDA_SH"
  echo "Set Q23_CONDA_SH to your conda.sh path and retry."
  exit 1
fi

conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo
echo "===== Running strategies ====="
echo

python -u run_strategy.py

echo
echo "===== Strategy runs complete ====="
echo

python -u run_dashboard.py || true

echo
echo "=== All tasks finished ==="
echo "Press any key to close this window..."
read -n 1 -s
