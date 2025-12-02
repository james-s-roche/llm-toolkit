#!/usr/bin/env bash
# Lightweight launcher that ensures the local package is importable when running Streamlit
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Ensure the repository root is importable when Streamlit resets sys.path[0]
export PYTHONPATH="$ROOT_DIR":${PYTHONPATH:-}

streamlit run "$ROOT_DIR/webui/streamlit_app.py"



