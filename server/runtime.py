# ================================================================================
# runtime.py
#
# GPU / モデル / 参照音声の実行時状態。
# ================================================================================

import contextvars
import gc
import os
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from server.gpu_runtime import (
    configure_backend_env,
    disable_rocm_jit_fuser,
    print_torch_device,
    print_vram_usage,
    reclaim_cuda_memory,
    resolve_torch_device,
)

# ROCm の最適化。torch を import する前に設定すること。CUDA では無視される。
configure_backend_env()

from server.miopen_log import flush_miopen_warnings, install_miopen_log_filter

install_miopen_log_filter()

import torch

disable_rocm_jit_fuser(torch)

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    resolve_runtime_device,
)

from server.dilated_conv import install_dense_dilated_conv
from server.model_store import (
    ensure_codec_weights,
    ensure_model_checkpoint,
    require_model,
    resolve_model_id,
)
from server.sampling.prepare_voice import load_voice_wav
from server.sampling.voice_store import find_voice, read_catalog


device = None
precision = None
tts = None
active_model_id = None
active_checkpoint_path = None
voice_lock = threading.Lock()
active_ref_wav = None
active_ref_latent = None
active_voice_id = None
_loaded = False

# SamplingRequest 既定に合わせる（参照エンコードのキャッシュ用）。
_REF_NORMALIZE_DB = -16.0
_REF_ENSURE_MAX = True


# リクエストごとの透かし指定。合成は同じスレッドで同期的に走る。
watermark_request = contextvars.ContextVar("irodori_watermark", default=False)


class _OptionalWatermarker:
    """リクエストが透かしを求めたときだけ SilentCipher を通す。"""

    def __init__(self, inner):
        self.inner = inner

    @property
    def ready(self):
        if not watermark_request.get():
            return True

        return bool(getattr(self.inner, "ready", False))

    def encode_batch(self, audios, *, sample_rate):
        if not watermark_request.get():
            return audios

        if self.inner is None or not getattr(self.inner, "ready", False):
            return audios

        return self.inner.encode_batch(audios, sample_rate=sample_rate)


# ================================================================================
# apply_voice
# メモリ上の参照 WAV を、指定の声へ載せ替える。
# ================================================================================
def apply_voice(voice):
    global active_ref_wav
    global active_ref_latent
    global active_voice_id

    with voice_lock:
        if voice is None:
            active_ref_wav = None
            active_ref_latent = None
            active_voice_id = None
            return None

        wav_path = load_voice_wav(voice)
        active_ref_wav = wav_path
        active_voice_id = voice.get("id") if wav_path else None
        active_ref_latent = (
            _ensure_ref_latent_cache(wav_path) if wav_path and tts is not None else None
        )
        return wav_path


# ================================================================================
# _ref_latent_cache_path
# 参照 WAV に対応する latent キャッシュパスを返す。
# ================================================================================
def _ref_latent_cache_path(wav_path):
    return wav_path + ".irodori_ref_latent.pt"


# ================================================================================
# _load_wav_tensor
# 参照 WAV を (C, T) テンソルで読む。soundfile 優先、失敗時は torchaudio。
# ================================================================================
def _load_wav_tensor(wav_path):
    try:
        import soundfile as sf

        data, sample_rate = sf.read(wav_path, dtype="float32")
        wav = torch.from_numpy(np.asarray(data, dtype=np.float32))
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        return wav, int(sample_rate)
    except Exception as soundfile_error:
        try:
            import torchaudio

            return torchaudio.load(wav_path)
        except Exception as torchaudio_error:
            raise RuntimeError(
                f"soundfile: {soundfile_error}; torchaudio: {torchaudio_error}"
            ) from torchaudio_error


# ================================================================================
# _ensure_ref_latent_cache
# 参照 WAV を一度だけ codec エンコードし、.pt に保存する。
# 毎文の GPU エンコードを避け、VRAM 断片化を抑える。
# ================================================================================
def _ensure_ref_latent_cache(wav_path):
    if tts is None or not wav_path:
        return None

    cache_path = _ref_latent_cache_path(wav_path)
    try:
        wav_mtime = os.path.getmtime(wav_path)
        if os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= wav_mtime:
            return cache_path
    except OSError:
        pass

    try:
        wav, sr = _load_wav_tensor(wav_path)
    except Exception as error:
        print(f"参照 latent キャッシュ失敗（WAV 読込）: {error}")
        return None

    max_ref_seconds = float(getattr(tts, "default_max_ref_seconds", 0) or 0)
    if max_ref_seconds > 0:
        max_samples = max(1, int(max_ref_seconds * float(sr)))
        if wav.shape[1] > max_samples:
            wav = wav[:, :max_samples]

    try:
        with torch.inference_mode():
            latent = tts.codec.encode_waveform(
                wav.unsqueeze(0),
                sample_rate=int(sr),
                normalize_db=_REF_NORMALIZE_DB,
                ensure_max=_REF_ENSURE_MAX,
            ).detach().cpu()
        if latent.ndim == 3 and latent.shape[0] == 1:
            latent = latent[0]
        torch.save(latent, cache_path)
    except Exception as error:
        print(f"参照 latent キャッシュ失敗（エンコード）: {error}")
        reclaim_vram()
        return None

    reclaim_vram()
    print(f"Reference latent cached: {cache_path}")
    return cache_path


# ================================================================================
# _park_silentcipher_on_cpu
# SilentCipher の重みを CPU に置く。GPU の VRAM は読み上げモデルに残す。
# ================================================================================
def _park_silentcipher_on_cpu(watermarker):
    model = getattr(watermarker, "model", None)

    if model is None:
        return

    device = str(getattr(model, "device", "cpu"))

    if device == "cpu" or device.startswith("cpu"):
        return

    model.device = "cpu"

    for name in ("enc_c", "dec_c", "stft"):
        module = getattr(model, name, None)

        if hasattr(module, "to"):
            setattr(model, name, module.to("cpu"))

    decoders = getattr(model, "dec_m", None)

    if isinstance(decoders, list):
        model.dec_m = [
            module.to("cpu") if hasattr(module, "to") else module
            for module in decoders
        ]


# ================================================================================
# _prepare_watermarker
# SilentCipher を CPU に残し、リクエストの watermark 指定で出し分ける。
# ================================================================================
def _prepare_watermarker():
    if tts is None:
        return

    inner = getattr(tts, "watermarker", None)

    if isinstance(inner, _OptionalWatermarker):
        inner = inner.inner

    _park_silentcipher_on_cpu(inner)
    tts.watermarker = _OptionalWatermarker(inner)
    reclaim_vram()

    if getattr(inner, "ready", False):
        print("SilentCipher watermark: loaded on CPU")
    else:
        print("SilentCipher watermark: unavailable")


# ================================================================================
# load_startup_voice
# 保存済みの active 声があれば参照 WAV を読む。無ければ参照なしで待受する。
# ================================================================================
def load_startup_voice():
    catalog = read_catalog()
    voice = find_voice(catalog, catalog.get("active_id"))

    if voice is None:
        print()
        print("参照音声は未選択です。既定のアニメ声で読み上げます。")
        apply_voice(None)
        return

    print()
    print(f"Loading reference voice: {voice.get('name') or voice.get('stem')}")

    prompt_start = time.perf_counter()
    wav_path = apply_voice(voice)

    if wav_path is None:
        print("参照 WAV がありません。拡張から声を選び直すか、追加してください。")
        return

    print(f"Reference voice loaded in {time.perf_counter() - prompt_start:.4f} sec")
    print(f"Reference wav: {wav_path}")
    if active_ref_latent:
        print(f"Reference latent: {active_ref_latent}")


# ================================================================================
# print_vram
# 現在の VRAM 使用量を出す。
# ================================================================================
def print_vram():
    print_vram_usage(torch)


# ================================================================================
# reclaim_vram
# 生成後などに CUDA キャッシュを返す。
# ================================================================================
def reclaim_vram():
    reclaim_cuda_memory(torch)


# ================================================================================
# _cpu_tensor_gb
# モデルが CPU に置いたままの重みの合計を返す。
# ================================================================================
def _cpu_tensor_gb(runtime):
    if runtime is None:
        return 0.0

    total = 0
    seen = set()

    modules = []

    for candidate in (getattr(runtime, "model", None), getattr(runtime, "codec", None)):
        if candidate is None:
            continue

        if hasattr(candidate, "parameters") and hasattr(candidate, "buffers"):
            modules.append(candidate)
            continue

        inner = getattr(candidate, "model", None)

        if inner is not None and hasattr(inner, "parameters") and hasattr(inner, "buffers"):
            modules.append(inner)

    for module in modules:
        tensors = list(module.parameters())
        tensors.extend(module.buffers())

        for tensor in tensors:
            if tensor is None or tensor.device.type != "cpu":
                continue

            try:
                storage = tensor.untyped_storage()
                pointer = storage.data_ptr()
                nbytes = storage.nbytes()
            except Exception:
                continue

            if pointer == 0 or pointer in seen:
                continue

            seen.add(pointer)
            total += nbytes

    return total / (1024 ** 3)


# ================================================================================
# _runtime_precision
# GPU なら bf16、CPU なら fp32 を返す。
# ================================================================================
def _runtime_precision(resolved_device):
    if resolved_device.get("backend") == "CPU":
        return "fp32"

    return "bf16"


# ================================================================================
# _build_runtime
# 指定精度で InferenceRuntime を作る。
# ================================================================================
def _build_runtime(checkpoint_path, codec_path, model_device, precision):
    key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=model_device,
        codec_repo=codec_path,
        model_precision=precision,
        codec_device=model_device,
        codec_precision=precision,
    )
    target = resolve_runtime_device(model_device)

    # CUDA でも ROCm でもデバイス型は cuda。7900 XTX と 780M もここを通る。
    if Path(checkpoint_path).suffix.lower() == ".safetensors" and target.type == "cuda":
        from server.stream_weights import build_streaming_runtime

        return build_streaming_runtime(key)

    return InferenceRuntime.from_key(key)


# ================================================================================
# _release_tts
# 読み込み済みの重みを外し、VRAM を返す。
# ================================================================================
def _release_tts():
    global tts

    previous = tts
    tts = None

    if previous is None:
        return

    previous.model = None
    previous.codec = None
    previous.watermarker = None
    del previous
    gc.collect()
    reclaim_vram()


# ================================================================================
# _effective_precision
# CPU では bf16 を使えないので、指定が bf16 でも fp32 を返す。
# ================================================================================
def _effective_precision(requested):
    chosen = requested or precision

    if chosen == "bf16" and device.get("backend") == "CPU":
        return "fp32"

    return chosen


# ================================================================================
# _load_tts
# 指定精度で読み、失敗時は bf16 から fp32 へ落とす。
# 量子化チェックポイントの bf16 指定は、GPU のときだけ固定する。
# ================================================================================
def _load_tts(checkpoint_path, codec_path, model_precision=None):
    global tts
    global precision

    chosen = _effective_precision(model_precision)
    allow_fp32_fallback = model_precision is None and chosen == "bf16"
    model_device = device["device_map"]

    try:
        tts = _build_runtime(
            checkpoint_path,
            codec_path,
            model_device,
            chosen,
        )
    except Exception as error:
        if not allow_fp32_fallback:
            print(f"Failed to load Irodori-TTS: {error}")
            traceback.print_exc()
            _write_load_error(error)
            raise

        print(f"bf16 load failed ({error}). Falling back to fp32...")
        precision = "fp32"
        chosen = "fp32"

        try:
            tts = _build_runtime(
                checkpoint_path,
                codec_path,
                model_device,
                chosen,
            )
        except Exception as fallback_error:
            print(f"Failed to load Irodori-TTS: {fallback_error}")
            traceback.print_exc()
            _write_load_error(fallback_error)
            raise

    print(f"Using precision: {chosen}")


# ================================================================================
# _activate_loaded_model
# いまの重みを外してから、指定チェックポイントを読み込む。
# ================================================================================
def _activate_loaded_model(spec, checkpoint_path, codec_path):
    global active_model_id
    global active_checkpoint_path

    requested_precision = spec.get("precision") or precision
    load_precision = _effective_precision(requested_precision)

    print(f"Loading {spec['name']}...")
    print(f"Model path: {checkpoint_path}")
    print(f"Codec path: {codec_path}")

    if load_precision != requested_precision:
        print("CPU では bf16 を使えないため、fp32 で読みます。")

    print(f"Precision: {load_precision}")

    _release_tts()
    load_start = time.perf_counter()
    _load_tts(checkpoint_path, codec_path, spec.get("precision"))
    _prepare_watermarker()
    flush_miopen_warnings()
    # チェックポイントは一度メインメモリへ展開してから GPU へコピーする。
    # コピー元はここでもう使わない。
    reclaim_vram()
    cpu_gb = _cpu_tensor_gb(tts)
    active_model_id = spec["id"]
    active_checkpoint_path = checkpoint_path
    print(f"Model loaded in {time.perf_counter() - load_start:.2f} sec")

    if cpu_gb >= 0.5:
        print(f"CPU weights: {cpu_gb:.2f} GB")

    print_vram()


# ================================================================================
# switch_model
# モデルをダウンロードし、読み上げ中の生成が終わってから切り替える。
# ================================================================================
def switch_model(model_id):
    spec = require_model(model_id)

    if spec["id"] == active_model_id and tts is not None:
        return spec

    checkpoint_path = ensure_model_checkpoint(spec)
    codec_path = ensure_codec_weights()

    with voice_lock:
        if spec["id"] == active_model_id and tts is not None:
            return spec

        previous_id = active_model_id
        previous_checkpoint = active_checkpoint_path
        print()
        print(f"Switching model: {spec['name']}")

        try:
            _activate_loaded_model(spec, checkpoint_path, codec_path)
        except Exception:
            if previous_id and previous_checkpoint:
                print(f"Restoring model: {previous_id}")

                try:
                    _activate_loaded_model(
                        require_model(previous_id),
                        previous_checkpoint,
                        codec_path,
                    )
                except Exception as restore_error:
                    print(f"Failed to restore previous model: {restore_error}")

            raise

    return spec


# ================================================================================
# _write_load_error
# モデル読み込み失敗を server_error.log へ残す。
# ================================================================================
def _write_load_error(error):
    try:
        with open("server_error.log", "w", encoding="utf-8") as handle:
            handle.write(f"Failed to load Irodori-TTS: {error}\n\n")
            traceback.print_exc(file=handle)
    except OSError:
        pass


# ================================================================================
# load_runtime
# デバイス判定、Irodori-TTS の読み込み、起動時の声ロードを行う。
# ================================================================================
def load_runtime():
    global device
    global precision
    global _loaded

    if _loaded:
        return

    from server.extension_settings import read_extension_settings

    device = resolve_torch_device(torch)
    precision = _runtime_precision(device)
    model_device = device["device_map"]
    spec = require_model(resolve_model_id(read_extension_settings().get("model")))

    print("=" * 60)
    print(f"Irodori Voice Reader Server - {device['label']}")
    print("=" * 60)

    print()
    print_torch_device(torch, device)

    print()
    checkpoint_path = ensure_model_checkpoint(spec)
    codec_path = ensure_codec_weights()
    install_dense_dilated_conv(torch, model_device)
    _activate_loaded_model(spec, checkpoint_path, codec_path)

    load_startup_voice()
    _loaded = True


# ================================================================================
