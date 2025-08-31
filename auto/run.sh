#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Change directory to the parent directory (one level up from where the script is)
cd "$SCRIPT_DIR/.." || { 
    echo "Error: Failed to change directory to parent."
    exit 1
}

# --- Crucial Check: Ensure auto/__init__.py exists ---
# Python needs this file to recognize 'auto' as a package for '-m' to work.
if [ ! -f "auto/__init__.py" ]; then
    echo "Warning: auto/__init__.py not found. Creating it."
    # Attempt to create it. Handle potential permission errors gracefully.
    touch "auto/__init__.py" || {
        echo "Error: Failed to create auto/__init__.py. Please create it manually."
        exit 1
    }
fi
# --- End Check ---

# Execute the python module 'auto.main', passing along all arguments ($@)
# given to this shell script.
echo "Executing: python -m auto.main $@"
python -m auto.main "$@"

# Optional: Capture the exit code from python and exit the script with it
# exit $?
