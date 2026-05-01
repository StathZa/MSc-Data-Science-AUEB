#!/bin/bash
echo "Installing packages for automatic library detection and project handling"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Master script detected at: $SCRIPT_DIR"
echo "$PROJECT_DIR"

pip install uv
uv pip install pipreqs

pipreqs "$PROJECT_DIR" --savepath "$SCRIPT_DIR/requirements.txt" --force

# Pin versions compatible with Colab
sed -i 's/numpy==.*/numpy>=2.0,<2.1/' "$SCRIPT_DIR/requirements.txt"
sed -i 's/datasets==.*/datasets>=2.18,<3/' "$SCRIPT_DIR/requirements.txt"
sed -i 's/pandas==.*/pandas>=2.0,<3/' "$SCRIPT_DIR/requirements.txt"
sed -i 's/protobuf==.*/protobuf>=5.29,<7/' "$SCRIPT_DIR/requirements.txt"
sed -i 's/\+cpu//' "$SCRIPT_DIR/requirements.txt"

uv pip install -r "$SCRIPT_DIR/requirements.txt"

# Upgrade fsspec
pip install --upgrade fsspec

if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    uv init "$PROJECT_DIR"
fi

uv add -r "$SCRIPT_DIR/requirements.txt" --frozen