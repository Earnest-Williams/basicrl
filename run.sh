#!/bin/bash
set -euo pipefail

# Determine paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

# --- Check for active conda/mamba environment ---
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "WARNING: No Conda/Mamba environment appears to be active."
else
  ENV_NAME="$(basename "$CONDA_PREFIX")"
  if [ "$ENV_NAME" != "$PROJECT_NAME" ]; then
    echo "WARNING: Active environment '$ENV_NAME' != project '$PROJECT_NAME'"
  fi
fi

# --- Run the game ---
echo "Running: python main.py $@"
cd "$PROJECT_ROOT"
python main.py "$@"
