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

# If already running, just open the browser
if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${PORT}/"; then
    xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 &
    exit 0
fi

# Otherwise start in background, wait until reachable, then open browser
LOG="${HOME}/.nvfp4_ui.log"
nohup "$PY" "$HERE/nvfp4_ui.py" > "$LOG" 2>&1 &
echo "  PID=$! · log: $LOG"
for i in $(seq 1 30); do
    if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${PORT}/"; then
        xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 &
        exit 0
    fi
    sleep 1
done
echo "[nvfp4_ui] did not come up in 30s, see $LOG"
tail -n 20 "$LOG"
exit 1
