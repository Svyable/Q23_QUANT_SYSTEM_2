#!/bin/bash
# Q23: Launch dashboard only (double-click friendly on macOS)
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

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
else
  echo "Could not find conda init script at $CONDA_SH"
  exit 1
fi

conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -u run_dashboard.py

echo
echo "Press any key to close this window..."
read -n 1 -s
