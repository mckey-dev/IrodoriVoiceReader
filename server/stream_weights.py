# ================================================================================
# stream_weights.py
#
# チェックポイントを1テンソルずつ GPU へ置く。
# ROCm（7900 XTX / 780M）も CUDA も、PyTorch では cuda デバイスとして同じ経路を使う。
# ================================================================================

import gc
from pathlib import Path

import torch
from safetensors import safe_open

from irodori_tts.codec import DACVAECodec
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    _CONFIG_META_KEY,
    _TEXT_ENCODER_CONFIG_META_KEY,
    _default_max_ref_seconds,
    _maybe_compile_inference_model,
    _move_inference_module,
    _parse_json_mapping,
    _resolve_tokenizer_source,
    _split_flat_checkpoint_config,
    resolve_runtime_device,
    resolve_runtime_dtype,
)
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.quantization import (
    parse_quantization_metadata,
    unflatten_quantized_state_dict,
)
from irodori_tts.tokenizer import PretrainedTextTokenizer

from server.gpu_runtime import _windows_ram_gb


# ================================================================================
# _note_peak_ram
# 転送中にプロセスが使ったメインメモリの最大値を覚える。
# ================================================================================
def _note_peak_ram(peak_gb):
    usage = _windows_ram_gb()

    if usage is None:
        return peak_gb

    return max(peak_gb, usage["process"])


# ================================================================================
# _read_tensor
# ファイルを開いて1枚だけ読み、GPU へ写してから閉じる。
# 開きっぱなしにすると、触ったページがプロセスに残る。
# ================================================================================
def _read_tensor(path, name, device, dtype):
    peak_gb = 0.0

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        cpu_tensor = handle.get_tensor(name)
        peak_gb = _note_peak_ram(peak_gb)

        if cpu_tensor.is_floating_point() and dtype is not None and cpu_tensor.dtype != dtype:
            placed = cpu_tensor.to(device=device, dtype=dtype)
        else:
            placed = cpu_tensor.to(device=device)

        del cpu_tensor
        peak_gb = _note_peak_ram(peak_gb)

    return placed, _note_peak_ram(peak_gb)


# ================================================================================
# _assign_parameter
# meta 上のパラメータを、GPU 上の実テンソルで置き換える。
# ================================================================================
def _assign_parameter(model, name, tensor):
    parent = model
    parts = name.split(".")

    for part in parts[:-1]:
        parent = getattr(parent, part)

    leaf = parts[-1]
    current = getattr(parent, leaf)

    if tuple(current.shape) != tuple(tensor.shape):
        raise RuntimeError(
            f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)}, model {tuple(current.shape)}."
        )

    parent.register_parameter(
        leaf,
        torch.nn.Parameter(tensor, requires_grad=False),
    )


# ================================================================================
# _materialize_buffers
# チェックポイントに無いバッファを、計算し直して GPU に置く。
# meta のままでは RoPE の値が消えている。
# ================================================================================
def _is_rope_embedding(module):
    return (
        hasattr(module, "layer_types")
        and hasattr(module, "rope_type")
        and hasattr(module, "compute_default_rope_parameters")
        and hasattr(module, "config")
    )


def _materialize_buffers(model, device):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    for module in model.modules():
        # v4 Large は T5Gemma2、v4.1 Anime は ModernBERT。どちらも層種別の inv_freq を持つ。
        if _is_rope_embedding(module):
            for layer_type in module.rope_type:
                rope_init_fn = module.compute_default_rope_parameters

                if module.rope_type.get(layer_type) != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]

                inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                inv_freq = inv_freq.detach().to(device=device, dtype=torch.float32)
                module.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
                module.register_buffer(
                    f"{layer_type}_original_inv_freq",
                    inv_freq.clone(),
                    persistent=False,
                )

        if hasattr(module, "scalar_embed_scale"):
            scale = torch.tensor(
                float(module.scalar_embed_scale),
                dtype=torch.float32,
                device=device,
            )
            module.register_buffer("embed_scale", scale, persistent=False)

        if "_freqs_cis_cache" in getattr(module, "_buffers", {}):
            module.register_buffer(
                "_freqs_cis_cache",
                torch.empty(0, 0, dtype=torch.complex64, device=device),
                persistent=False,
            )

    leftover = []

    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is None or not getattr(buf, "is_meta", False):
                continue

            if buf.numel() == 0:
                module.register_buffer(
                    name,
                    torch.empty(tuple(buf.shape), dtype=buf.dtype, device=device),
                    persistent=False,
                )
                continue

            leftover.append(f"{module.__class__.__name__}.{name}")

    if leftover:
        raise RuntimeError(
            "Checkpoint left meta buffers without values: " + ", ".join(leftover[:6])
        )


# ================================================================================
# _stream_plain_weights
# 通常チェックポイントの重みを、パラメータ名ごとに GPU へ置く。
# ================================================================================
def _stream_plain_weights(model, path, device, dtype):
    names = [name for name, _ in model.named_parameters()]
    peak_gb = 0.0

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        file_keys = set(handle.keys())

    missing = [name for name in names if name not in file_keys]
    unexpected = sorted(file_keys.difference(names))

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not match the model: "
            f"missing {missing[:4]}, unexpected {unexpected[:4]}."
        )

    for name in names:
        placed, seen_gb = _read_tensor(path, name, device, dtype)
        peak_gb = max(peak_gb, seen_gb)
        _assign_parameter(model, name, placed)
        del placed

    return peak_gb


# ================================================================================
# _stream_quantized_weights
# 量子化チェックポイントは展開用のキーがパラメータ名と違う。
# 1枚ずつ GPU へ置いてから、torchao の形に戻して割り当てる。
# ================================================================================
def _stream_quantized_weights(model, path, metadata, device):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        file_keys = list(handle.keys())

    peak_gb = 0.0
    flattened = {}

    for name in file_keys:
        placed, seen_gb = _read_tensor(path, name, device, dtype=None)
        peak_gb = max(peak_gb, seen_gb)
        flattened[name] = placed

    state_dict, _payload = unflatten_quantized_state_dict(flattened, metadata=metadata)
    del flattened
    model.load_state_dict(state_dict, assign=True, strict=True)
    del state_dict
    return peak_gb


# ================================================================================
# _empty_model
# 重みの実体を持たないモデルを作る。先に fp32 を確保しない。
# ================================================================================
def _empty_model(model_cfg, text_encoder_config):
    with torch.device("meta"):
        return TextToLatentRFDiT(
            model_cfg,
            pretrained_backbone_config=text_encoder_config,
            load_pretrained_backbone_weights=not model_cfg.use_pretrained_text_encoder,
        )


# ================================================================================
# build_streaming_runtime
# safetensors を小分けに GPU へ載せて InferenceRuntime を返す。
# ================================================================================
def build_streaming_runtime(key: RuntimeKey) -> InferenceRuntime:
    model_device = resolve_runtime_device(key.model_device)
    codec_device = resolve_runtime_device(key.codec_device)
    model_dtype = resolve_runtime_dtype(
        precision=key.model_precision,
        device=model_device,
    )
    codec_dtype = resolve_runtime_dtype(
        precision=key.codec_precision,
        device=codec_device,
    )
    checkpoint_path = Path(key.checkpoint)

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    flat_config = _parse_json_mapping(
        metadata.get(_CONFIG_META_KEY),
        field=_CONFIG_META_KEY,
        path=checkpoint_path,
        required=True,
    )
    text_encoder_config = _parse_json_mapping(
        metadata.get(_TEXT_ENCODER_CONFIG_META_KEY),
        field=_TEXT_ENCODER_CONFIG_META_KEY,
        path=checkpoint_path,
    )
    model_cfg_dict, inference_cfg = _split_flat_checkpoint_config(
        path=checkpoint_path,
        flat_config=flat_config,
    )
    model_cfg = merge_dataclass_overrides(
        ModelConfig(),
        model_cfg_dict,
        section="checkpoint model_config",
    )
    quantization = parse_quantization_metadata(metadata)

    print(
        f"Streaming weights to {model_device} ({str(model_dtype).removeprefix('torch.')}), "
        "one tensor at a time."
    )

    model = _empty_model(model_cfg, text_encoder_config)

    if quantization is None:
        peak_gb = _stream_plain_weights(model, checkpoint_path, model_device, model_dtype)
    else:
        peak_gb = _stream_quantized_weights(model, checkpoint_path, metadata, model_device)

    _materialize_buffers(model, model_device)
    model = _move_inference_module(model, device=model_device, dtype=model_dtype)
    model.eval()
    model = _maybe_compile_inference_model(
        model,
        enabled=bool(key.compile_model),
        dynamic=bool(key.compile_dynamic),
    )

    meta_left = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if getattr(tensor, "is_meta", False)
    ]

    if meta_left:
        raise RuntimeError(f"Weights were not placed on {model_device}: {meta_left[:4]}")

    gc.collect()

    if peak_gb > 0:
        print(f"Weight stream peak RAM: {peak_gb:.2f} GB")

    text_tokenizer_source, text_tokenizer_is_local = _resolve_tokenizer_source(
        checkpoint_path,
        model_cfg.text_tokenizer_repo,
    )
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=text_tokenizer_source,
        add_bos=bool(model_cfg.text_add_bos),
        local_files_only=text_tokenizer_is_local,
        revision=None if text_tokenizer_is_local else model_cfg.text_encoder_revision,
    )

    if (
        not model_cfg.use_pretrained_text_encoder
        and tokenizer.vocab_size != model_cfg.text_vocab_size
    ):
        raise ValueError(
            f"text_vocab_size mismatch: checkpoint text_vocab_size={model_cfg.text_vocab_size} but tokenizer "
            f"({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
        )

    caption_tokenizer = None

    if model_cfg.use_caption_condition:
        caption_tokenizer_source, caption_tokenizer_is_local = _resolve_tokenizer_source(
            checkpoint_path,
            model_cfg.caption_tokenizer_repo_resolved,
        )
        caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=caption_tokenizer_source,
            add_bos=model_cfg.caption_add_bos_resolved,
            local_files_only=caption_tokenizer_is_local,
            revision=(
                None if caption_tokenizer_is_local else model_cfg.text_encoder_revision
            ),
        )

        if (
            not model_cfg.use_pretrained_text_encoder
            and caption_tokenizer.vocab_size != model_cfg.caption_vocab_size_resolved
        ):
            raise ValueError(
                "caption_vocab_size mismatch: "
                f"checkpoint caption_vocab_size={model_cfg.caption_vocab_size_resolved} but tokenizer "
                f"({model_cfg.caption_tokenizer_repo_resolved}) vocab_size={caption_tokenizer.vocab_size}."
            )

    default_text_max_len = 256
    default_caption_max_len = default_text_max_len
    default_max_ref_seconds = _default_max_ref_seconds(inference_cfg)

    if isinstance(inference_cfg, dict):
        ckpt_text_max_len = inference_cfg.get("max_text_len")

        if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
            default_text_max_len = int(ckpt_text_max_len)

        ckpt_caption_max_len = inference_cfg.get("max_caption_len")

        if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
            default_caption_max_len = int(ckpt_caption_max_len)
        else:
            default_caption_max_len = default_text_max_len

    codec = DACVAECodec.load(
        repo_id=key.codec_repo,
        device=str(codec_device),
        dtype=codec_dtype,
        deterministic_encode=bool(key.codec_deterministic_encode),
        deterministic_decode=bool(key.codec_deterministic_decode),
    )

    if model_cfg.latent_dim != codec.latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: checkpoint latent_dim={model_cfg.latent_dim} "
            f"but codec latent_dim={codec.latent_dim}."
        )

    return InferenceRuntime(
        key=key,
        model_cfg=model_cfg,
        train_cfg=inference_cfg if isinstance(inference_cfg, dict) else None,
        model=model,
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        codec=codec,
        default_text_max_len=default_text_max_len,
        default_caption_max_len=default_caption_max_len,
        default_max_ref_seconds=default_max_ref_seconds,
    )
