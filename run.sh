#!/bin/bash
set -euo pipefail

# Determine paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"

# --- Check for active conda/mamba environment ---
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "WARNING: No Conda/Mamba environment appears to be active."
else
  ENV_NAME="$(basename "$CONDA_PREFIX")"
  if [ "$ENV_NAME" != "$PROJECT_NAME" ]; then
    echo "WARNING: Active environment '$ENV_NAME' != project '$PROJECT_NAME'"
  fi
fi

# --- Run main as a module ---
echo "Running: python -m $PROJECT_NAME.main $@"
# Ensure project modules resolve without fallback imports
PYTHONPATH="$PARENT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  python -m "$PROJECT_NAME.main" "$@"
