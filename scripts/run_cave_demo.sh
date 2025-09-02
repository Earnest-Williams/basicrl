#!/usr/bin/env bash

# Install dependencies before running:
#   pip install -r requirements.txt

set -euo pipefail

# Determine project root from this script's location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$PROJECT_ROOT"

# Expose repo to Python
export PYTHONPATH="$(pwd)"

python main.py --log-level INFO "$@"
