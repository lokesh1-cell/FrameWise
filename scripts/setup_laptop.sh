#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  SELECTED_PYTHON=""
  for candidate in "${PYTHON:-python3}" "$HOME/anaconda3/bin/python"; do
    if "$candidate" -c 'import tkinter; import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
      SELECTED_PYTHON="$candidate"
      break
    fi
  done
  if [[ -z "$SELECTED_PYTHON" ]]; then
    echo 'Python 3.11+ with Tkinter is required. Install your distribution’s Python Tk package and retry.' >&2
    exit 1
  fi
  "$SELECTED_PYTHON" -m venv .venv
fi
if ! .venv/bin/python -c 'import tkinter' 2>/dev/null; then
  echo 'The existing .venv interpreter has no Tkinter. Install Tk for that Python interpreter, or create .venv with a Tk-enabled Python.' >&2
  exit 1
fi
.venv/bin/python -m pip install -e .
echo 'Ready. Launch with: bash run_ui.sh'
