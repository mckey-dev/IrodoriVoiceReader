#!/usr/bin/env bash

# ================================================================================
# start_amd.sh
#
# uv の ROCm extra で Irodori-TTS サーバーを起動する。
# 公式 extra は Linux / WSL 向けです。
# ================================================================================

set -euo pipefail

cd "$(dirname "$0")"
export CK_LOGGING=0
export UV_PROJECT_ENVIRONMENT="${PWD}/venv_amd"

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

echo "依存パッケージを同期しています: --extra rocm （venv_amd）"
${UV} sync --extra rocm

echo "サーバーを起動します"
exec ${UV} run --no-sync python -m server "$@"

# ================================================================================
