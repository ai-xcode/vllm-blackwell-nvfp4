#!/bin/bash
# launch_nvfp4_ui.sh — start the NVFP4 Conversion Center UI.
#
# Default port: 8770. Override with NVFP4_UI_PORT.
# Default python: $PYTHON or python3.
#
# Required: a python with `nicegui` installed and able to import
# `subprocess`, `asyncio`. Conversion needs the venv at $NVFP4_CONVERT_VENV
# (default ~/nvfp4_conversion/venv); serving needs $NVFP4_SERVE_VENV
# (default ~/vLLM_Servers/vllm_env).

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${NVFP4_UI_PORT:-8770}"
PY="${PYTHON:-python3}"

if ! "$PY" -c "import nicegui" 2>/dev/null; then
    echo "ERROR: nicegui not importable from '$PY'" >&2
    echo "       try: $PY -m pip install nicegui" >&2
    exit 1
fi

echo "[nvfp4_ui] http://127.0.0.1:${PORT}"
exec "$PY" "$HERE/nvfp4_ui.py"
