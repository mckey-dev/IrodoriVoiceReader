# ================================================================================
# extension_settings.py
#
# 拡張の設定ファイル extension/settings.json の形と読み書き。
# ================================================================================

import json
import os
import tempfile

from pydantic import BaseModel, Field

from server.config import (
    DEFAULT_CAPTION,
    DEFAULT_CFG_SCALE_CAPTION,
    DEFAULT_CFG_SCALE_SPEAKER,
    DEFAULT_CFG_SCALE_TEXT,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_STEPS,
    DEFAULT_SPEED,
    DEFAULT_SWAY_COEFF,
    DEFAULT_T_SCHEDULE_MODE,
    DEFAULT_WATERMARK,
    SETTINGS_FILE,
)
from server.generation import normalize_schedule_mode
from server.model_store import resolve_model_id


# ================================================================================
# ExtensionSettings
# 拡張の設定ファイル extension/settings.json の形。
# ================================================================================
class ExtensionSettings(BaseModel):
    model: str = DEFAULT_MODEL_ID
    playbackSpeed: float = Field(default=1.0, ge=0.5, le=2.0)
    playbackVolume: float = Field(default=1.0, ge=0, le=1)
    sentenceGapMs: int = Field(default=500, ge=0, le=2000)
    caption: str = DEFAULT_CAPTION
    num_steps: int = Field(default=DEFAULT_NUM_STEPS, ge=1)
    speed: float = Field(default=DEFAULT_SPEED, gt=0)
    cfg_scale_text: float = Field(default=DEFAULT_CFG_SCALE_TEXT, ge=0)
    cfg_scale_caption: float = Field(default=DEFAULT_CFG_SCALE_CAPTION, ge=0)
    cfg_scale_speaker: float = Field(default=DEFAULT_CFG_SCALE_SPEAKER, ge=0)
    t_schedule_mode: str = DEFAULT_T_SCHEDULE_MODE
    sway_coeff: float = DEFAULT_SWAY_COEFF
    watermark: bool = DEFAULT_WATERMARK
    highlightSentence: bool = False
    autoReadChat: bool = False


# ================================================================================
# settings_to_dict
# Pydantic モデルを JSON 用の dict にする。
# ================================================================================
def settings_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


# ================================================================================
# default_extension_settings
# 拡張設定の既定値を返す。
# ================================================================================
def default_extension_settings():
    return settings_to_dict(ExtensionSettings())


# ================================================================================
# parse_extension_settings
# 設定 JSON を検証してモデルにする。
# ================================================================================
def parse_extension_settings(data):
    if hasattr(ExtensionSettings, "model_validate"):
        settings = ExtensionSettings.model_validate(data)
    else:
        settings = ExtensionSettings.parse_obj(data)

    payload = settings_to_dict(settings)
    payload["model"] = resolve_model_id(payload.get("model"))
    payload["t_schedule_mode"] = normalize_schedule_mode(
        payload.get("t_schedule_mode")
    )

    if hasattr(ExtensionSettings, "model_validate"):
        return ExtensionSettings.model_validate(payload)

    return ExtensionSettings.parse_obj(payload)


# ================================================================================
# stored_model_id
# settings.json に保存されているモデル ID を返す。
# ================================================================================
def stored_model_id():
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_MODEL_ID

    try:
        with open(SETTINGS_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return DEFAULT_MODEL_ID

    if not isinstance(data, dict):
        return DEFAULT_MODEL_ID

    return resolve_model_id(data.get("model"))


# ================================================================================
# write_extension_settings
# 拡張設定を settings.json に保存する。
# モデル ID は model_id を渡したときだけ更新する。
# ================================================================================
def write_extension_settings(settings, model_id=None):
    payload = dict(settings) if isinstance(settings, dict) else settings
    chosen = resolve_model_id(model_id) if model_id is not None else stored_model_id()

    if isinstance(payload, dict):
        payload = {
            key: value
            for key, value in payload.items()
            if key != "model"
        }

    normalized = settings_to_dict(parse_extension_settings(payload))
    normalized["model"] = chosen
    directory = os.path.dirname(SETTINGS_FILE)
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix="settings.",
        suffix=".tmp",
        dir=directory,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        os.replace(tmp_path, SETTINGS_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        raise

    return normalized


# ================================================================================
# read_extension_settings
# 拡張設定を settings.json から読む。無ければ作成する。
# ================================================================================
def read_extension_settings():
    if not os.path.exists(SETTINGS_FILE):
        return write_extension_settings(default_extension_settings())

    try:
        with open(SETTINGS_FILE, encoding="utf-8") as handle:
            data = json.load(handle)

        return settings_to_dict(parse_extension_settings(data))
    except Exception as error:
        print("Failed to read extension settings:", error)
        return default_extension_settings()


# ================================================================================
