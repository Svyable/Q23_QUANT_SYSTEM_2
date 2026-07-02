#!/bin/bash
# Smart Cryptocurrency Data Ingestion
# Double-click this file to run crypto database pull intelligently

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Export for use in script
export PROJECT_ROOT

# Load .env file first (for conda path and env name)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Load conda if available (adjust path if needed)
CONDA_SH="${Q23_CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
fi

# Activate conda environment if specified in .env
ENV_NAME="${Q23_CONDA_ENV:-qntdev}"
if [ ! -z "$ENV_NAME" ]; then
    conda activate "$ENV_NAME" 2>/dev/null || echo "Note: Could not activate conda env $ENV_NAME"
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Run the smart crypto ingestion script
echo "=========================================="
echo "Smart Cryptocurrency Data Ingestion"
echo "=========================================="
echo ""
python scripts/smart_ingest_crypto.py "$@"

# Keep terminal open to see results
echo ""
echo "=========================================="
echo "Press any key to close..."
read -n 1 -s
exit 0
