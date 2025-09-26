#!/usr/bin/env bash
set -euo pipefail

# Ensure pyenv Python is active
PY=$(pyenv which python || true)
if [[ -z "$PY" ]]; then
  echo "pyenv python not found. Did you install 3.12.3?"
  exit 1
fi

# Create venv inside repo
"$PY" -m venv .venv
source .venv/bin/activate

# Upgrade pip/setuptools/wheel
python -m pip install --upgrade pip setuptools wheel

# Install dev extras from pyproject
python -m pip install -e ".[dev]"

# Prepare pre-commit
pre-commit install

echo "✅ Setup complete. Activate with: source .venv/bin/activate"