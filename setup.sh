#!/usr/bin/env bash
# Linux/macOS one-time setup:  bash setup.sh
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python tools/fetch_models.py || true
python tools/selfcheck.py
echo "Activate later with:  source .venv/bin/activate"
