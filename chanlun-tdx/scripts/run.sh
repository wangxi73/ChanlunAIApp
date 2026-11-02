#!/bin/bash
# A wrapper script to automatically use the virtual environment

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SKILL_DIR="$(dirname "$DIR")"
VENV_DIR="$SKILL_DIR/.venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "Installing dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SKILL_DIR/requirements.txt"
fi

# Run the analysis script using the venv's Python
"$VENV_DIR/bin/python" "$DIR/analyze.py" "$@"
