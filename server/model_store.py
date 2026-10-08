# ================================================================================
# model_store.py
#
# Irodori-TTS と DACVAE コーデックを、リポジトリの models 配下へ置く。
# ================================================================================

import os

from huggingface_hub import hf_hub_download, snapshot_download

from server.config import DEFAULT_MODEL_ID
from server.paths import PROJECT_ROOT

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_DIR = os.path.join(PROJECT_ROOT, "models", "Semantic-DACVAE-Japanese-32dim")

# 同じ 32 次元 DACVAE を使う公開チェックポイント。
# v4 Large の量子化は、共通リポジトリのサブフォルダ。
_LARGE_QUANT_REPO = "Aratako/Irodori-TTS-v4-Large-Quantized"
_LARGE_QUANT_DIR = "Irodori-TTS-v4-Large-Quantized"
_GEMMA_NOTE = "Gemma の利用条件が適用される。"

TTS_MODELS = (
    {
        "id": "v4.1-anime",
        "name": "Irodori-TTS-v4.1-Anime",
        "label": "v4.1 Anime（0.8B）",
        "repo": "phasefield-audio/Irodori-TTS-v4.1-Anime",
        "note": "アニメ調の読み上げ向け。",
    },
    {
        "id": "v4-small",
        "name": "Irodori-TTS-v4-Small",
        "label": "v4 Small（0.8B）",
        "repo": "Aratako/Irodori-TTS-v4-Small",
        "note": "公式の v4 Small。",
    },
    {
        "id": "v4.1-small",
        "name": "Irodori-TTS-v4.1-Small",
        "label": "v4.1 Small（0.8B）",
        "repo": "Aratako/Irodori-TTS-v4.1-Small",
        "note": "公式の v4.1 Small。長さの予測を改善した版。",
    },
    {
        "id": "v4-large",
        "name": "Irodori-TTS-v4-Large",
        "label": "v4 Large（量子化なし・約 13GB）",
        "repo": "Aratako/Irodori-TTS-v4-Large",
        "note": "公式のフル精度 v4 Large。量子化していない通常の重み。初回は約 13GB。Gemma の利用条件が適用される。",
    },
    {
        "id": "v4-large-int8",
        "name": "Irodori-TTS-v4-Large-int8-weight-only",
        "dir": _LARGE_QUANT_DIR,
        "label": "v4 Large INT8（3.6GB）",
        "repo": _LARGE_QUANT_REPO,
        "subfolder": "int8-weight-only",
        "precision": "bf16",
        "note": "重みのみ INT8 の v4 Large。CUDA 向けの汎用版。" + _GEMMA_NOTE,
    },
    {
        "id": "v4-large-int8-dynamic",
        "name": "Irodori-TTS-v4-Large-int8-dynamic",
        "dir": _LARGE_QUANT_DIR,
        "label": "v4 Large INT8 dynamic（3.6GB）",
        "repo": _LARGE_QUANT_REPO,
        "subfolder": "int8-dynamic",
        "precision": "bf16",
        "note": "重みと活性化を INT8 にした v4 Large。CUDA 向け。" + _GEMMA_NOTE,
    },
    {
        "id": "v4-large-int4",
        "name": "Irodori-TTS-v4-Large-int4-weight-only",
        "dir": _LARGE_QUANT_DIR,
        "label": "v4 Large INT4（2.8GB）",
        "repo": _LARGE_QUANT_REPO,
        "subfolder": "int4-weight-only",
        "precision": "bf16",
        "note": "最も小さい INT4 の v4 Large。NVIDIA Ampere 以降。" + _GEMMA_NOTE,
    },
    {
        "id": "v4-large-fp8",
        "name": "Irodori-TTS-v4-Large-float8-weight-only",
        "dir": _LARGE_QUANT_DIR,
        "label": "v4 Large FP8（3.7GB）",
        "repo": _LARGE_QUANT_REPO,
        "subfolder": "float8-weight-only",
        "precision": "bf16",
        "note": "重みのみ FP8 の v4 Large。NVIDIA Ada 以降。" + _GEMMA_NOTE,
    },
    {
        "id": "v4-large-fp8-dynamic",
        "name": "Irodori-TTS-v4-Large-float8-dynamic",
        "dir": _LARGE_QUANT_DIR,
        "label": "v4 Large FP8 dynamic（3.7GB）",
        "repo": _LARGE_QUANT_REPO,
        "subfolder": "float8-dynamic",
        "precision": "bf16",
        "note": "重みと活性化を FP8 にした v4 Large。NVIDIA Ada 以降。" + _GEMMA_NOTE,
    },
)

MODEL_BY_ID = {spec["id"]: spec for spec in TTS_MODELS}


# ================================================================================
# _model_dir
# models/<リポジトリ名> を返す。量子化版はバリアントでフォルダを共有する。
# ================================================================================
def _model_dir(spec):
    folder = spec.get("dir") or spec["name"]
    return os.path.join(PROJECT_ROOT, "models", folder)


# ================================================================================
# _checkpoint_file
# 読み込む model.safetensors のパスを返す。
# ================================================================================
def _checkpoint_file(spec):
    parts = [_model_dir(spec)]
    subfolder = spec.get("subfolder")

    if subfolder:
        parts.append(subfolder)

    parts.append("model.safetensors")
    return os.path.join(*parts)


# ================================================================================
# _tokenizer_config
# チェックポイントと同じモデルディレクトリの tokenizer 設定を返す。
# ================================================================================
def _tokenizer_config(spec):
    return os.path.join(_model_dir(spec), "tokenizer", "tokenizer_config.json")


# ================================================================================
# _is_model_ready
# チェックポイントと tokenizer があるかを返す。
# ================================================================================
def _is_model_ready(spec):
    return (
        os.path.isfile(_checkpoint_file(spec))
        and os.path.isfile(_tokenizer_config(spec))
    )


# ================================================================================
# resolve_model_id
# 知っているモデル ID ならそれを、無ければ既定を返す。
# ================================================================================
def resolve_model_id(model_id):
    key = str(model_id or "").strip()

    if key in MODEL_BY_ID:
        return key

    return DEFAULT_MODEL_ID


# ================================================================================
# require_model
# モデル ID の定義を返す。未知なら KeyError。
# ================================================================================
def require_model(model_id):
    key = str(model_id or "").strip()
    spec = MODEL_BY_ID.get(key)

    if spec is None:
        raise KeyError(key)

    return spec


# ================================================================================
# public_model_catalog
# 操作画面へ渡すモデル一覧。
# ================================================================================
def public_model_catalog():
    return [
        {
            "id": spec["id"],
            "name": spec["name"],
            "label": spec["label"],
            "repo": spec["repo"],
            "note": spec["note"],
        }
        for spec in TTS_MODELS
    ]


# ================================================================================
# ensure_model_checkpoint
# 指定モデルの model.safetensors を返す。無ければ Hugging Face から取る。
# ================================================================================
def ensure_model_checkpoint(spec):
    checkpoint_path = _checkpoint_file(spec)

    if _is_model_ready(spec):
        return checkpoint_path

    model_dir = _model_dir(spec)
    os.makedirs(model_dir, exist_ok=True)
    subfolder = spec.get("subfolder")

    if subfolder:
        source = spec["repo"] + "/" + subfolder
        allow_patterns = [
            subfolder + "/model.safetensors",
            "tokenizer/*",
        ]
    else:
        source = spec["repo"]
        allow_patterns = ["model.safetensors", "tokenizer/*"]

    print(f"Downloading {source} to {model_dir} ...")

    snapshot_download(
        repo_id=spec["repo"],
        local_dir=model_dir,
        allow_patterns=allow_patterns,
    )

    if not _is_model_ready(spec):
        raise FileNotFoundError(
            f"Model was not downloaded to {checkpoint_path}"
        )

    return checkpoint_path


# ================================================================================
# ensure_codec_weights
# models/Semantic-DACVAE-Japanese-32dim/weights.pth を返す。
# ================================================================================
def ensure_codec_weights():
    weights_path = os.path.join(CODEC_DIR, "weights.pth")

    if os.path.isfile(weights_path):
        return weights_path

    os.makedirs(CODEC_DIR, exist_ok=True)

    print(f"Downloading {CODEC_REPO} to {CODEC_DIR} ...")

    try:
        snapshot_download(
            repo_id=CODEC_REPO,
            local_dir=CODEC_DIR,
            allow_patterns=["weights.pth"],
        )
    except Exception:
        downloaded = hf_hub_download(
            repo_id=CODEC_REPO,
            filename="weights.pth",
            local_dir=CODEC_DIR,
        )
        if os.path.abspath(downloaded) != os.path.abspath(weights_path):
            raise FileNotFoundError(
                f"Codec weights were downloaded to {downloaded}"
            )

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Codec was not downloaded to {weights_path}"
        )

    return weights_path

# ================================================================================
