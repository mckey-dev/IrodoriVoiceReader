# ================================================================================
# app.py
#
# Irodori-TTS をローカル GPU（ROCm / CUDA）または CPU で動かす TTS サーバー。
# HTTP ルート。モデルと声の実行時状態は runtime が持つ。
# ================================================================================

import io
import os
import shutil
import time
import traceback

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

import soundfile as sf
import torch

from irodori_tts.inference_runtime import SamplingRequest, resolve_cfg_scales

from server import runtime
from server.config import APP_VERSION, MAX_VOICE_UPLOAD_BYTES, SETTINGS_FILE
from server.extension_settings import (
    ExtensionSettings,
    read_extension_settings,
    settings_to_dict,
    write_extension_settings,
)
from server.generation import (
    TTSRequest,
    default_generation_options,
    duration_scale_from_speed,
    resolve_generation_options,
)
from server.miopen_log import flush_miopen_warnings
from server.model_store import public_model_catalog, require_model
from server.paths import AUDIO_EXTENSIONS
from server.runtime import apply_voice, load_runtime
from server.sampling.prepare_voice import prepare_voice_wav
from server.sampling.voice_store import (
    NONE_VOICE_ID,
    add_voice,
    delete_voice,
    extension_of,
    find_voice,
    new_voice_id,
    public_catalog,
    read_catalog,
    sanitize_stem,
    set_active,
    voice_paths,
)


load_runtime()


# FastAPI の初期化

app = FastAPI(
    title="Irodori Voice Reader",
    version=APP_VERSION,
)


# CORS 設定

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================================
# _audio_to_wav
# 生成テンソルを WAV バイト列にする。
# ================================================================================
def _audio_to_wav(audio_tensor, sample_rate):
    audio_cpu = audio_tensor.detach().to(device="cpu", dtype=torch.float32)

    if audio_cpu.ndim == 2:
        if audio_cpu.shape[0] == 1:
            audio_np = audio_cpu.squeeze(0).numpy()
        elif audio_cpu.shape[1] == 1:
            audio_np = audio_cpu.squeeze(1).numpy()
        else:
            audio_np = audio_cpu.mean(dim=0).numpy()
    else:
        audio_np = audio_cpu.numpy()

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_np, int(sample_rate), format="WAV")
    return wav_buffer.getvalue(), int(audio_np.shape[-1])


# ルート

# ================================================================================
# root
# サーバーの稼働確認用情報を返す。
# ================================================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Irodori Voice Reader",
        "device": runtime.device["label"],
        "model": _active_model_name(),
    }


# ================================================================================
# _active_model_name
# 読み込み中のチェックポイント名を返す。
# ================================================================================
def _active_model_name():
    try:
        return require_model(runtime.active_model_id)["name"]
    except KeyError:
        return None


# ================================================================================
# _model_state
# 選択できるモデルと、いま読み込んでいる ID を返す。
# ================================================================================
def _model_state():
    return {
        "status": "ok",
        "active": runtime.active_model_id,
        "name": _active_model_name(),
        "models": public_model_catalog(),
    }


# ================================================================================
# health
# ヘルスチェック用の簡易応答を返す。
# ================================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
    }


# ================================================================================
# tts_defaults
# 生成オプションの既定値を返す。クライアントの初期表示用。
# ================================================================================
@app.get("/tts/defaults")
def tts_defaults():
    return {
        "status": "ok",
        "defaults": default_generation_options(),
    }


# ================================================================================
# get_extension_settings
# 拡張の設定ファイルを返す。
# ================================================================================
@app.get("/extension/settings")
def get_extension_settings():
    return {
        "status": "ok",
        "path": SETTINGS_FILE,
        "settings": read_extension_settings(),
    }


# ================================================================================
# put_extension_settings
# 拡張の設定ファイルを保存する。
# ================================================================================
@app.put("/extension/settings")
def put_extension_settings(settings: ExtensionSettings):
    saved = write_extension_settings(settings_to_dict(settings))

    return {
        "status": "ok",
        "path": SETTINGS_FILE,
        "settings": saved,
    }


# ================================================================================
# get_models
# 切り替えられるモデルと、いま読み込んでいるモデルを返す。
# ================================================================================
@app.get("/models")
def get_models():
    return _model_state()


# ================================================================================
# select_model
# チェックポイントを切り替え、次回起動用に settings.json へ残す。
# 未ダウンロードならここで取得する。
# ================================================================================
@app.put("/models/{model_id}")
def select_model(model_id: str):
    try:
        require_model(model_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="指定したモデルが見つかりません。",
        )

    try:
        runtime.switch_model(model_id)
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"モデルの読み込みに失敗しました: {error}",
        )

    write_extension_settings(
        read_extension_settings(),
        model_id=model_id,
    )
    return _model_state()


# ================================================================================
# list_voices
# 登録済みの声一覧と、今使う声を返す。
# ================================================================================
@app.get("/voices")
def list_voices():
    return public_catalog(read_catalog())


# ================================================================================
# create_voice
# 参照音声を保存し、WAV にして active にする。
# ================================================================================
@app.post("/voices")
def create_voice(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    filename = file.filename or "voice.wav"
    source_ext = extension_of(filename)

    if not source_ext:
        allowed = ", ".join(AUDIO_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"対応していない形式です。{allowed} を指定してください。",
        )

    payload = file.file.read()

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="ファイルが空です。",
        )

    if len(payload) > MAX_VOICE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail="ファイルが大きすぎます（20MB まで）。",
        )

    stem = sanitize_stem(filename)
    display_name = (name or "").strip() or stem
    voice_id = new_voice_id()
    voice = {
        "id": voice_id,
        "name": display_name,
        "stem": stem,
        "source_name": os.path.basename(filename),
        "source_ext": source_ext,
        "has_wav": False,
    }
    paths = voice_paths(voice)
    os.makedirs(paths["dir"], exist_ok=True)

    try:
        with open(paths["source"], "wb") as handle:
            handle.write(payload)

        print()
        print(f"Saving voice: {display_name}")
        prepare_voice_wav(voice)
        voice["has_wav"] = True
        catalog = add_voice(voice, make_active=True)
    except Exception as error:
        traceback.print_exc()

        if os.path.isdir(paths["dir"]):
            shutil.rmtree(paths["dir"], ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"声の準備に失敗しました: {error}",
        )

    apply_voice(voice)
    return public_catalog(catalog)


# ================================================================================
# select_voice
# 使う声を切り替える。none なら参照なし。
# ================================================================================
@app.put("/voices/{voice_id}/select")
def select_voice(voice_id: str):
    catalog = read_catalog()

    if voice_id == NONE_VOICE_ID:
        _voice, catalog = set_active(None)
        apply_voice(None)
        return public_catalog(catalog)

    voice = find_voice(catalog, voice_id)

    if voice is None:
        raise HTTPException(
            status_code=404,
            detail="指定した声が見つかりません。",
        )

    paths = voice_paths(voice)

    if not os.path.isfile(paths["wav"]) and not os.path.isfile(paths["source"]):
        raise HTTPException(
            status_code=409,
            detail="この声の参照音声がありません。もう一度アップロードしてください。",
        )

    voice, catalog = set_active(voice_id)
    apply_voice(voice)
    return public_catalog(catalog)


# ================================================================================
# remove_voice
# 声とその中間ファイルを削除する。
# ================================================================================
@app.delete("/voices/{voice_id}")
def remove_voice(voice_id: str):
    catalog = read_catalog()

    if find_voice(catalog, voice_id) is None:
        raise HTTPException(
            status_code=404,
            detail="指定した声が見つかりません。",
        )

    _removed, catalog = delete_voice(voice_id)
    next_voice = find_voice(catalog, catalog.get("active_id"))
    apply_voice(next_voice)
    return public_catalog(catalog)


# ================================================================================
# tts
# テキストから音声を生成し、WAV を返す。
# ================================================================================
@app.post("/tts")
def tts(request: TTSRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="text is empty",
        )

    options = resolve_generation_options(request)
    duration_scale = duration_scale_from_speed(options["speed"])

    # 生成中はモデルを切り替えない。声の付け替えもこの文が終わるまで待つ。
    with runtime.voice_lock:
        if runtime.tts is None:
            raise HTTPException(
                status_code=503,
                detail="モデルが読み込まれていません。",
            )

        ref_wav = runtime.active_ref_wav
        ref_latent = runtime.active_ref_latent
        use_caption = bool(options["caption"])
        use_speaker = bool(ref_latent or ref_wav)

        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = (
            resolve_cfg_scales(
                cfg_guidance_mode="independent",
                cfg_scale_text=float(options["cfg_scale_text"]),
                cfg_scale_caption=float(options["cfg_scale_caption"]),
                cfg_scale_speaker=float(options["cfg_scale_speaker"]),
                cfg_scale=None,
                use_caption_condition=use_caption,
                use_speaker_condition=use_speaker,
            )
        )

        print()
        print("-" * 60)
        print("TTS request:")
        if request.request_id:
            print(f"id: {request.request_id}")
        print(f"chars: {len(text)}")
        print(f"model: {_active_model_name() or runtime.active_model_id}")
        print(f"voice: {runtime.active_voice_id or 'none'}")
        print(text)
        print("Options:", options)
        print(f"duration_scale: {duration_scale:.4f}")
        for message in scale_messages:
            print(message)

        start = time.perf_counter()
        watermark_token = runtime.watermark_request.set(bool(options["watermark"]))

        try:
            result = runtime.tts.synthesize(
                SamplingRequest(
                    text=text,
                    caption=options["caption"],
                    # latent キャッシュがあれば WAV 再エンコードを避ける
                    ref_wav=None if ref_latent else ref_wav,
                    ref_latent=ref_latent,
                    no_ref=not use_speaker,
                    seconds=None,
                    duration_scale=duration_scale,
                    num_steps=options["num_steps"],
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_caption=cfg_scale_caption,
                    cfg_scale_speaker=cfg_scale_speaker,
                    t_schedule_mode=options["t_schedule_mode"],
                    sway_coeff=float(options["sway_coeff"]),
                ),
                log_fn=None,
            )

        except Exception as e:

            print("Generation error:")
            print(e)
            traceback.print_exc()
            runtime.reclaim_vram()

            raise HTTPException(
                status_code=500,
                detail=str(e),
            )
        finally:
            runtime.watermark_request.reset(watermark_token)

    elapsed = time.perf_counter() - start

    flush_miopen_warnings()

    wav_data, samples = _audio_to_wav(result.audio, result.sample_rate)
    used_seed = result.used_seed
    sample_rate = result.sample_rate
    del result
    runtime.reclaim_vram()

    duration = samples / sample_rate if samples and sample_rate else 0

    print(f"Generation finished in {elapsed:.2f} sec")
    print(f"Audio: {samples} samples, {duration:.2f} sec")
    print(f"seed: {used_seed}")
    runtime.print_vram()

    if samples <= 0:
        print("Warning: generated audio is empty")

    print(f"Audio size: {len(wav_data)} bytes")

    return Response(
        content=wav_data,
        media_type="audio/wav",
    )


# ================================================================================
