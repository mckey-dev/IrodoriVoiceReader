# ================================================================================
# config.py
#
# サーバーの定数。HOST / PORT は python -m server --host / --port で上書きできる。
# ================================================================================

import os

from server.paths import PROJECT_ROOT


APP_VERSION = "1.0.0"
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "extension", "settings.json")
MAX_VOICE_UPLOAD_BYTES = 20 * 1024 * 1024

HOST = "127.0.0.1"
PORT = 8000

# 起動時に読むチェックポイント。操作画面から切り替えられる。
DEFAULT_MODEL_ID = "v4.1-anime"

# Irodori-TTS の生成既定値。
# speed は話す速さ（大きいほど短い）。内部では duration_scale = 1 / speed にする。
DEFAULT_CAPTION = ""
DEFAULT_NUM_STEPS = 40
DEFAULT_SPEED = 1.0
DEFAULT_CFG_SCALE_TEXT = 3.0
DEFAULT_CFG_SCALE_CAPTION = 3.0
DEFAULT_CFG_SCALE_SPEAKER = 5.0
DEFAULT_T_SCHEDULE_MODE = "linear"
DEFAULT_SWAY_COEFF = -1.0
DEFAULT_WATERMARK = False

# ================================================================================
