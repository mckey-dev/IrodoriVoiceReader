# ================================================================================
# paths.py
#
# リポジトリ直下を基準にしたパス。
# ================================================================================

import os


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

# 声クローン用の参照音声。コードではなく実行時データ。
VOICE_DIR = os.path.join(PROJECT_ROOT, "voice")
VOICES_FILE = os.path.join(VOICE_DIR, "voices.json")

AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".ogg",
    ".oga",
    ".flac",
)

# ================================================================================
