#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change directory to the project root (two levels up from the script)
cd "$PROJECT_ROOT" || {
    echo "Error: Failed to change directory to project root."
    exit 1
}

# --- Crucial Check: Ensure prototypes/auto/__init__.py exists ---
# Python needs this file to recognize 'prototypes.auto' as a package for '-m' to work.
if [ ! -f "prototypes/auto/__init__.py" ]; then
    echo "Warning: prototypes/auto/__init__.py not found. Creating it."
    # Attempt to create it. Handle potential permission errors gracefully.
    touch "prototypes/auto/__init__.py" || {
        echo "Error: Failed to create prototypes/auto/__init__.py. Please create it manually."
        exit 1
    }
fi
# --- End Check ---

# Execute the python module 'prototypes.auto.main', passing along all arguments ($@)
# given to this shell script.
echo "Executing: python -m prototypes.auto.main $@"
python -m prototypes.auto.main "$@"

# Optional: Capture the exit code from python and exit the script with it
# exit $?
