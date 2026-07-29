#!/usr/bin/env sh

CURRENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$CURRENT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 0.0.0.0 只能表示“监听所有网卡”，不适合作为浏览器访问地址。
# macOS/Linux 下浏览器打开 http://0.0.0.0:8501 可能会经过代理或网关，
# 最终出现 502。默认绑定并打开 127.0.0.1，与 Windows 启动脚本保持一致。
IA2_WEBUI_HOST="${IA2_WEBUI_HOST:-${MPT_WEBUI_HOST:-127.0.0.1}}"
IA2_WEBUI_PORT="${IA2_WEBUI_PORT:-${MPT_WEBUI_PORT:-8501}}"

if [ -x "$CURRENT_DIR/.venv/bin/python" ]; then
  PORT_CHECK_CMD="$CURRENT_DIR/.venv/bin/python"
  set -- "$CURRENT_DIR/.venv/bin/python" -m streamlit
elif command -v uv >/dev/null 2>&1; then
  PORT_CHECK_CMD="uv run python"
  set -- uv run streamlit
elif command -v streamlit >/dev/null 2>&1; then
  echo "***** Warning: using streamlit from PATH. If dependencies fail, run 'uv sync --frozen' first. *****"
  PORT_CHECK_CMD="python3"
  set -- streamlit
else
  echo "***** Neither project Python, uv, nor streamlit was found. Please install dependencies first. *****"
  exit 1
fi

find_available_port() {
  WEBUI_HOST="$IA2_WEBUI_HOST" WEBUI_PORT="$IA2_WEBUI_PORT" "$@" - <<'PY' 2>/dev/null
import os
import socket
import sys

host = os.environ.get("WEBUI_HOST", "127.0.0.1")
preferred = int(os.environ.get("WEBUI_PORT", "8501"))
candidates = [preferred] + [port for port in range(8502, 8600) if port != preferred]

for port in candidates:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            continue
        print(port)
        sys.exit(0)

sys.exit(1)
PY
}

# 用 Python 做端口探测，避免依赖 lsof/nc 在不同 macOS/Linux 发行版上的差异。
# shellcheck disable=SC2086
SELECTED_WEBUI_PORT=$(find_available_port $PORT_CHECK_CMD)

if [ -z "$SELECTED_WEBUI_PORT" ]; then
  echo "***** No available WebUI port found in 8501-8599 for $IA2_WEBUI_HOST. *****"
  exit 1
fi

if [ "$SELECTED_WEBUI_PORT" != "$IA2_WEBUI_PORT" ]; then
  echo "***** Port $IA2_WEBUI_PORT is unavailable, using $SELECTED_WEBUI_PORT instead. *****"
fi

IA2_WEBUI_PORT="$SELECTED_WEBUI_PORT"

echo "***** WebUI address: http://$IA2_WEBUI_HOST:$IA2_WEBUI_PORT *****"
"$@" run "$CURRENT_DIR/webui/Main.py" \
  --server.address="$IA2_WEBUI_HOST" \
  --server.port="$IA2_WEBUI_PORT" \
  --browser.serverAddress="$IA2_WEBUI_HOST" \
  --browser.gatherUsageStats=False \
  --client.toolbarMode=minimal \
  --logger.hideWelcomeMessage=True \
  --server.showEmailPrompt=False \
  --server.enableCORS=True
