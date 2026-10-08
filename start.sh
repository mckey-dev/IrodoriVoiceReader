#!/usr/bin/env bash

# ================================================================================
# start.sh
#
# uv の CPU extra で Irodori-TTS サーバーを起動する。
# ================================================================================

set -euo pipefail

cd "$(dirname "$0")"
export UV_PROJECT_ENVIRONMENT="${PWD}/venv"

if command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "Python が見つかりません。Python 3 をインストールしてください。"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV="uv"
elif "${PY}" -m uv --version >/dev/null 2>&1; then
    UV="${PY} -m uv"
else
    echo "uv をインストールしています..."
    "${PY}" -m pip install -U uv
    UV="${PY} -m uv"
fi

echo "依存パッケージを同期しています: --extra cpu （venv）"
${UV} sync --extra cpu

echo "サーバーを起動します"
exec ${UV} run --no-sync python -m server "$@"

# ================================================================================
