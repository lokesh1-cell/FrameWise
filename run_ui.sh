#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run bash scripts/setup_laptop.sh first.' >&2
  exit 1
fi
if ! .venv/bin/python -c 'import tkinter; import av; import cv2; import pydantic_settings' 2>/dev/null; then
  echo 'The desktop dependencies are missing. Run bash scripts/setup_laptop.sh with a Python installation that includes Tkinter.' >&2
  exit 1
fi
exec .venv/bin/python -m physical_ai_pipeline.desktop
