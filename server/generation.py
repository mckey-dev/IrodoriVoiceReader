# ================================================================================
# generation.py
#
# /tts の入力モデルと、生成オプションの既定値合成。
# ================================================================================

from typing import Optional

from pydantic import BaseModel, Field

from server.config import (
    DEFAULT_CAPTION,
    DEFAULT_CFG_SCALE_CAPTION,
    DEFAULT_CFG_SCALE_SPEAKER,
    DEFAULT_CFG_SCALE_TEXT,
    DEFAULT_NUM_STEPS,
    DEFAULT_SPEED,
    DEFAULT_SWAY_COEFF,
    DEFAULT_T_SCHEDULE_MODE,
    DEFAULT_WATERMARK,
)


# ================================================================================
# TTSRequest
# /tts に送る読み上げテキストと、任意の生成オプションを受け取る。
# 省略した項目はサーバー既定値を使う。
# ================================================================================
class TTSRequest(BaseModel):
    text: str
    request_id: Optional[str] = None
    caption: Optional[str] = None
    num_steps: Optional[int] = Field(default=None, ge=1)
    speed: Optional[float] = Field(default=None, gt=0)
    cfg_scale_text: Optional[float] = Field(default=None, ge=0)
    cfg_scale_caption: Optional[float] = Field(default=None, ge=0)
    cfg_scale_speaker: Optional[float] = Field(default=None, ge=0)
    t_schedule_mode: Optional[str] = None
    sway_coeff: Optional[float] = None
    watermark: Optional[bool] = None


# ================================================================================
# normalize_schedule_mode
# linear / sway 以外は linear に戻す。
# ================================================================================
def normalize_schedule_mode(value):
    mode = str(value or "").strip().lower()

    if mode == "sway":
        return "sway"

    return DEFAULT_T_SCHEDULE_MODE


# ================================================================================
# default_generation_options
# サーバー側の生成既定値を返す。
# ================================================================================
def default_generation_options():
    return {
        "caption": DEFAULT_CAPTION,
        "num_steps": DEFAULT_NUM_STEPS,
        "speed": DEFAULT_SPEED,
        "cfg_scale_text": DEFAULT_CFG_SCALE_TEXT,
        "cfg_scale_caption": DEFAULT_CFG_SCALE_CAPTION,
        "cfg_scale_speaker": DEFAULT_CFG_SCALE_SPEAKER,
        "t_schedule_mode": DEFAULT_T_SCHEDULE_MODE,
        "sway_coeff": DEFAULT_SWAY_COEFF,
        "watermark": DEFAULT_WATERMARK,
    }


# ================================================================================
# resolve_generation_options
# リクエストの指定値と既定値を合成する。空の caption は未指定として扱う。
# ================================================================================
def resolve_generation_options(request):
    caption = (
        request.caption.strip()
        if isinstance(request.caption, str)
        else ""
    )

    return {
        "caption": caption or None,
        "num_steps": (
            DEFAULT_NUM_STEPS
            if request.num_steps is None
            else request.num_steps
        ),
        "speed": (
            DEFAULT_SPEED
            if request.speed is None
            else request.speed
        ),
        "cfg_scale_text": (
            DEFAULT_CFG_SCALE_TEXT
            if request.cfg_scale_text is None
            else request.cfg_scale_text
        ),
        "cfg_scale_caption": (
            DEFAULT_CFG_SCALE_CAPTION
            if request.cfg_scale_caption is None
            else request.cfg_scale_caption
        ),
        "cfg_scale_speaker": (
            DEFAULT_CFG_SCALE_SPEAKER
            if request.cfg_scale_speaker is None
            else request.cfg_scale_speaker
        ),
        "t_schedule_mode": normalize_schedule_mode(request.t_schedule_mode),
        "sway_coeff": (
            DEFAULT_SWAY_COEFF
            if request.sway_coeff is None
            else request.sway_coeff
        ),
        "watermark": (
            DEFAULT_WATERMARK
            if request.watermark is None
            else bool(request.watermark)
        ),
    }


# ================================================================================
# duration_scale_from_speed
# 話す速さ speed を、Irodori の duration_scale に変換する。
# ================================================================================
def duration_scale_from_speed(speed):
    value = float(speed)

    if value <= 0:
        raise ValueError("speed must be greater than 0")

    return 1.0 / value

# ================================================================================
