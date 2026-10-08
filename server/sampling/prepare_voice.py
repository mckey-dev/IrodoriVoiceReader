# ================================================================================
# prepare_voice.py
#
# 1 件の声について、参照 WAV を用意する。
# ================================================================================

import os

from server.sampling.sample_audio import convert_to_wav
from server.sampling.voice_store import voice_paths


# ================================================================================
# load_voice_wav
# 参照 WAV のパスを返す。無ければ None。
# ================================================================================
def load_voice_wav(voice):
    if voice is None:
        return None

    paths = voice_paths(voice)
    wav_path = paths["wav"]

    if os.path.isfile(wav_path):
        return wav_path

    source_path = paths["source"]

    if os.path.isfile(source_path):
        return convert_to_wav(source_path, wav_path)

    return None


# ================================================================================
# prepare_voice_wav
# アップロードした音声をモノラル WAV にして返す。
# ================================================================================
def prepare_voice_wav(voice):
    paths = voice_paths(voice)
    source_path = paths["source"]
    wav_path = paths["wav"]

    if not os.path.isfile(source_path) and os.path.isfile(wav_path):
        source_path = wav_path

    if not os.path.isfile(source_path):
        raise FileNotFoundError("参照音声がありません。")

    return convert_to_wav(source_path, wav_path)

# ================================================================================
