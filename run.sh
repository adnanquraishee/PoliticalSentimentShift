#!/bin/bash

# Configuration
VENV_DIR="venv"
PYTHON_SCRIPT="main.py"
REQUIREMENTS_FILE="requirements.txt"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   Political Sentiment Shift Prediction - Setup & Run Script   ${NC}"
echo -e "${BLUE}================================================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}Creating virtual environment in $VENV_DIR...${NC}"
    python3 -m venv "$VENV_DIR"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Define paths to executables
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# Check if venv executables exist
if [ ! -x "$PIP" ] || [ ! -x "$PYTHON" ]; then
    echo "Error: Virtual environment executables not found in $VENV_DIR/bin/"
    exit 1
fi

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
"$PIP" install --upgrade pip

# Install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${GREEN}Installing dependencies from $REQUIREMENTS_FILE...${NC}"
    "$PIP" install -r "$REQUIREMENTS_FILE" || exit 1
else
    echo "Requirements file $REQUIREMENTS_FILE not found!"
    exit 1
fi

# Run the Python script
if [ -f "$PYTHON_SCRIPT" ]; then
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}   Running Project Pipeline...   ${NC}"
    echo -e "${BLUE}================================================================${NC}"
    # Set PYTHONPATH to include current directory
    export PYTHONPATH=$PYTHONPATH:.
    "$PYTHON" "$PYTHON_SCRIPT"
else
    echo "Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

echo -e "${GREEN}Done.${NC}"
