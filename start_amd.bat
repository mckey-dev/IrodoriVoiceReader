@echo off
setlocal EnableExtensions
cd /d "%~dp0"
>nul chcp 65001
@rem
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set CK_LOGGING=0
set mimalloc_purge_delay=0
set "UV_PROJECT_ENVIRONMENT=%~dp0venv_amd"

rem uv + Windows ROCm（gfx1100 / gfx1103）。仮想環境は venv_amd

set "PY="
where py >nul 2>&1
if errorlevel 1 goto TRY_PYTHON
set "PY=py -3"
goto HAVE_PY

:TRY_PYTHON
where python >nul 2>&1
if errorlevel 1 goto NO_PYTHON
set "PY=python"
goto HAVE_PY

:NO_PYTHON
echo Python が見つかりません。Python 3 をインストールしてください。
exit /b 1

:HAVE_PY
set "UV="
where uv >nul 2>&1
if not errorlevel 1 (
  set "UV=uv"
  goto HAVE_UV
)
%PY% -m uv --version >nul 2>&1
if not errorlevel 1 (
  set "UV=%PY% -m uv"
  goto HAVE_UV
)
echo uv をインストールしています...
%PY% -m pip install -U uv
if errorlevel 1 goto UV_FAIL
set "UV=%PY% -m uv"

:HAVE_UV
echo 依存パッケージを同期しています: --extra rocm （venv_amd）
%UV% sync --extra rocm
if errorlevel 1 goto SYNC_FAIL
echo サーバーを起動します
%UV% run --no-sync python -m server %*
exit /b %ERRORLEVEL%

:UV_FAIL
echo uv のインストールに失敗しました。https://docs.astral.sh/uv/ を参照してください。
exit /b 1

:SYNC_FAIL
echo uv sync --extra rocm に失敗しました。
echo Windows では AMD ROCm 7.14（gfx1100 / gfx1103）の wheel を使います。
exit /b 1
