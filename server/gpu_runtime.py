# ================================================================================
# gpu_runtime.py
#
# PyTorch が使えるデバイスを判定する。ROCm / CUDA / CPU に対応する。
# ================================================================================

import ctypes
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys


NVIDIA_DRIVER_MIN_MAJOR = 570
NVIDIA_DRIVER_URL = "https://www.nvidia.com/ja-jp/geforce/drivers/"
DEVICE_ENV = "IRODORI_DEVICE"
# gfx1100 のような個別アーキテクチャ。gfx11 のような族パッケージは含めない。
_SPECIFIC_GPU_ARCH = re.compile(r"gfx(?:\d{4}|\d{3}|\d{2}[a-z])\Z")


# ================================================================================
# disable_rocm_jit_fuser
# Windows ROCm では TorchScript の GPU 融合が、HIP の bf16 定義と衝突して
# NVRTC コンパイルに失敗する。融合を止め、個別カーネルで実行する。
# ================================================================================
def disable_rocm_jit_fuser(torch_module):
    if not getattr(torch_module.version, "hip", None):
        return

    torch_module._C._jit_override_can_fuse_on_gpu(False)
    torch_module._C._jit_set_texpr_fuser_enabled(False)


# ================================================================================
# configure_backend_env
# ROCm / CUDA 向けの環境変数を、torch の import 前に設定する。
# ================================================================================
def configure_backend_env():
    # PyTorch の CPU 確保は mimalloc が保持する。0 にすると、GPU へコピーし終えた
    # チェックポイントの fp32 を OS へ返す。未設定のままだとプロセスに残る。
    os.environ.setdefault("mimalloc_purge_delay", "0")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # 未設定だと gfx1100 の畳み込みがグリッド形状を stdout に出す。明示的な 0 で止める。
    os.environ["CK_LOGGING"] = "0"
    # Linux のみ有効。可変長 TTS での CUDA 断片化を抑える。
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ================================================================================
# nvidia_smi_info
# nvidia-smi からドライバ版と CUDA 版を読む。無ければ空の情報を返す。
# ================================================================================
def nvidia_smi_info():
    info = {
        "available": False,
        "name": None,
        "driver_version": None,
        "cuda_version": None,
        "driver_major": None,
    }

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return info

    try:
        gpu = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        line = gpu.stdout.strip().splitlines()[0]
        name, driver_version = [part.strip() for part in line.split(",", 1)]
        info["available"] = True
        info["name"] = name
        info["driver_version"] = driver_version
        major_text = driver_version.split(".", 1)[0]
        if major_text.isdigit():
            info["driver_major"] = int(major_text)
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return info

    try:
        smi = subprocess.run(
            [nvidia_smi],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for raw_line in smi.stdout.splitlines():
            marker = None
            if "CUDA UMD Version:" in raw_line:
                marker = "CUDA UMD Version:"
            elif "CUDA Version:" in raw_line:
                marker = "CUDA Version:"
            if not marker:
                continue
            info["cuda_version"] = raw_line.split(marker, 1)[1].split()[0]
            break
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass

    return info


# ================================================================================
# is_gpu_torch_build
# GPU 向けにビルドされた PyTorch かどうかを返す。
# ================================================================================
def is_gpu_torch_build(torch_module):
    return bool(
        getattr(torch_module.version, "cuda", None)
        or getattr(torch_module.version, "hip", None)
    )


# ================================================================================
# driver_too_old_message
# NVIDIA ドライバが不足しているときの案内文を返す。
# ================================================================================
def driver_too_old_message(info=None):
    info = info or nvidia_smi_info()
    driver = info.get("driver_version") or "不明"
    cuda_version = info.get("cuda_version") or "不明"
    name = info.get("name") or "NVIDIA GPU"

    return (
        f"{name} は検出されていますが、NVIDIA ドライバ {driver} "
        f"（CUDA {cuda_version}）では現行の PyTorch CUDA を使えません。\n"
        f"ドライバ {NVIDIA_DRIVER_MIN_MAJOR} 以降に更新してから、"
        f"start_cuda.bat を実行してください。\n"
        f"{NVIDIA_DRIVER_URL}"
    )


# ================================================================================
# require_nvidia_driver
# CUDA 起動に必要なドライバが入っているかを確認する。不足なら終了する。
# ================================================================================
def require_nvidia_driver():
    info = nvidia_smi_info()

    if not info["available"]:
        print("nvidia-smi が見つかりません。NVIDIA ドライバを入れてください。")
        print(NVIDIA_DRIVER_URL)
        raise SystemExit(1)

    print(f"NVIDIA GPU: {info['name']}")
    print(f"NVIDIA ドライバ: {info['driver_version']}")
    if info.get("cuda_version"):
        print(f"ドライバの CUDA: {info['cuda_version']}")

    if info["driver_major"] is None or info["driver_major"] < NVIDIA_DRIVER_MIN_MAJOR:
        print()
        print(driver_too_old_message(info))
        raise SystemExit(1)


# ================================================================================
# gpu_arch_name
# HIP の gcnArchName から、gfx1100 のようなアーキテクチャ名を取り出す。
# ================================================================================
def gpu_arch_name(raw):
    token = str(raw or "").strip().lower().split(":", 1)[0]
    if token.startswith("gfx"):
        return token
    return ""


# ================================================================================
# arch_from_device_package
# rocm-sdk-device-gfx1100 のようなパッケージ名から対象アーキテクチャを返す。
# amd-torch-device-gfx11 のような族パッケージは空文字。
# ================================================================================
def arch_from_device_package(name):
    lowered = str(name or "").strip().lower()
    marker = "device-"
    index = lowered.rfind(marker)
    if index < 0:
        return ""

    suffix = lowered[index + len(marker):]
    if _SPECIFIC_GPU_ARCH.fullmatch(suffix):
        return suffix
    return ""


# ================================================================================
# installed_rocm_device_archs
# 入っている ROCm のデバイスパッケージが対応するアーキテクチャを返す。
# 個別パッケージが無い Linux 公式 wheel では空。その場合は絞らない。
# ================================================================================
def installed_rocm_device_archs():
    archs = set()
    for dist in importlib.metadata.distributions():
        arch = arch_from_device_package(dist.metadata["Name"])
        if arch:
            archs.add(arch)
    return archs


# ================================================================================
# unsupported_gpu_reason
# 対象外 GPU を選ぶと HIP が落ちる理由を返す。
# ================================================================================
def unsupported_gpu_reason(device, supported_archs):
    arch = device.get("arch") or "不明"
    targets = "、".join(sorted(supported_archs)) or "不明"
    return (
        f"{device['name']}（{arch}）は、入っている ROCm（{targets}）の対象外です。\n"
        f"一覧に残る GPU か CPU を選んでください。\n"
        f"対象外の GPU へモデルを移すと、amdhip64 がアクセス違反で終了します。"
    )


# ================================================================================
# print_skipped_devices
# 選べない GPU を起動ログに出す。
# ================================================================================
def print_skipped_devices(devices):
    skipped = [device for device in devices if not device.get("supported", True)]
    if not skipped:
        return

    print()
    for device in skipped:
        print(device.get("skip_reason") or device["name"])
        print()


# ================================================================================
# _gpu_visibility_limited
# 利用者が既に GPU の可視範囲を絞っているかを返す。
# ================================================================================
def _gpu_visibility_limited():
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(key, "").strip():
            return True
    return False


# ================================================================================
# query_compute_devices
# このプロセスでは torch を import せず、子プロセスでデバイス一覧を取る。
# 内蔵 GPU を選んだあと、親が HIP_VISIBLE_DEVICES を書いてから torch を読むため。
# ================================================================================
def query_compute_devices():
    script = (
        "import json\n"
        "from server.gpu_runtime import (\n"
        "    configure_backend_env,\n"
        "    is_gpu_torch_build,\n"
        "    list_compute_devices,\n"
        ")\n"
        "configure_backend_env()\n"
        "import torch\n"
        "print(json.dumps({\n"
        "    'gpu_build': is_gpu_torch_build(torch),\n"
        "    'devices': list_compute_devices(torch),\n"
        "}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        raise SystemExit(proc.returncode or 1)

    line = ""
    for raw in proc.stdout.splitlines():
        if raw.strip():
            line = raw.strip()
    payload = json.loads(line)
    return payload["devices"], bool(payload["gpu_build"])


# ================================================================================
# use_integrated_gpu_alone
# 単体 GPU と内蔵 GPU が同時に見えると、内蔵 GPU の hipLaunchKernel が
# amdhip64 のアクセス違反で落ちる。選んだ内蔵 GPU だけを見せ、番号を cuda:0 にする。
# torch を import する前に呼ぶ。
# ================================================================================
def use_integrated_gpu_alone(choice, devices):
    if choice == "cpu" or _gpu_visibility_limited():
        return choice

    selected = next(device for device in devices if device["id"] == choice)
    if not selected.get("integrated"):
        return choice

    index = choice.split(":", 1)[1]
    os.environ["HIP_VISIBLE_DEVICES"] = index
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    os.environ["ROCR_VISIBLE_DEVICES"] = index
    print(f"{selected['name']} を単体の ROCm デバイスとして起動します。")
    return "cuda:0"


# ================================================================================
# cpu_device
# CPU 実行のデバイス情報を返す。
# ================================================================================
def cpu_device():
    return {
        "device_map": "cpu",
        "dtype": None,
        "name": "CPU",
        "backend": "CPU",
        "label": "CPU",
    }


# ================================================================================
# list_compute_devices
# 選べる GPU と CPU を返す。GPU が無いときは CPU だけ。
# ================================================================================
def list_compute_devices(torch_module):
    devices = []
    cuda = getattr(torch_module, "cuda", None)

    if cuda is not None and cuda.is_available():
        hip = bool(getattr(torch_module.version, "hip", None))
        backend = "ROCm" if hip else "CUDA"
        supported_archs = installed_rocm_device_archs() if hip else set()
        for index in range(cuda.device_count()):
            try:
                props = cuda.get_device_properties(index)
                name = props.name
                total_bytes = int(props.total_memory)
                integrated = bool(getattr(props, "is_integrated", 0))
                arch = gpu_arch_name(getattr(props, "gcnArchName", "")) if hip else ""
            except (RuntimeError, TypeError, ValueError, AttributeError):
                continue

            supported = arch in supported_archs if supported_archs else True
            device = {
                "id": f"cuda:{index}",
                "name": name,
                "backend": backend,
                "arch": arch,
                "integrated": integrated,
                "total_bytes": total_bytes,
                "supported": supported,
            }
            if not supported:
                device["skip_reason"] = unsupported_gpu_reason(device, supported_archs)
            devices.append(device)

    devices.append({
        "id": "cpu",
        "name": "CPU",
        "backend": "CPU",
        "arch": "",
        "integrated": False,
        "total_bytes": 0,
        "supported": True,
    })
    return devices


# ================================================================================
# default_device_id
# 単体 GPU のうち VRAM が最大のものを返す。GPU が無ければ cpu。
# ================================================================================
def default_device_id(devices):
    gpus = [
        device
        for device in devices
        if device["id"] != "cpu" and device.get("supported", True)
    ]
    discrete = [device for device in gpus if not device["integrated"]]
    pool = discrete or gpus

    if not pool:
        return "cpu"

    best = max(pool, key=lambda device: device["total_bytes"])
    return best["id"]


# ================================================================================
# normalize_device_choice
# cpu / cuda:N / GPU 番号を、一覧にある id にする。無ければ終了する。
# ================================================================================
def normalize_device_choice(choice, devices):
    text = str(choice).strip().lower()

    if text in ("", "auto"):
        return default_device_id(devices)

    if text == "cpu":
        return "cpu"

    if text.startswith("cuda:"):
        device_id = text
    elif text.isdigit():
        device_id = f"cuda:{int(text)}"
    else:
        device_id = text

    known = [device["id"] for device in devices]
    if device_id not in known:
        usable = [
            device["id"]
            for device in devices
            if device.get("supported", True)
        ]
        print(f"デバイス '{choice}' は使えません。選べるのは {', '.join(usable)} です。")
        raise SystemExit(1)

    selected = next(device for device in devices if device["id"] == device_id)
    if not selected.get("supported", True):
        print()
        print(selected.get("skip_reason") or f"{selected['name']} は使えません。")
        raise SystemExit(1)

    return device_id


# ================================================================================
# format_device_line
# 選択一覧の 1 行を返す。
# ================================================================================
def format_device_line(device):
    if device["id"] == "cpu":
        return "CPU"

    total_gb = device["total_bytes"] / (1024 ** 3)
    line = f"{device['backend']}  {device['name']}  {total_gb:.1f} GB"
    if device["integrated"]:
        line += "（内蔵）"
    return line


# ================================================================================
# resolve_torch_device
# 使えるデバイス、dtype、表示名を返す。
# IRODORI_DEVICE があればその指定を使い、無ければ単体 GPU を選ぶ。
# ================================================================================
def resolve_torch_device(torch_module):
    devices = list_compute_devices(torch_module)
    choice = os.environ.get(DEVICE_ENV, "").strip()

    if not choice or choice.lower() == "auto":
        if not any(device["id"] != "cpu" for device in devices):
            if is_gpu_torch_build(torch_module):
                print()
                print(driver_too_old_message())
                raise SystemExit(1)
            return _with_dtype(torch_module, cpu_device())
        choice = default_device_id(devices)
    else:
        choice = normalize_device_choice(choice, devices)

    if choice == "cpu":
        return _with_dtype(torch_module, cpu_device())

    selected = next(device for device in devices if device["id"] == choice)
    index = int(choice.split(":", 1)[1])
    torch_module.cuda.set_device(index)
    return _with_dtype(torch_module, {
        "device_map": choice,
        "dtype": None,
        "name": selected["name"],
        "backend": selected["backend"],
        "label": f"{selected['backend']} ({selected['name']})",
    })


# ================================================================================
# _with_dtype
# デバイス情報に、実行時の dtype を入れる。
# ================================================================================
def _with_dtype(torch_module, device):
    if device["device_map"] == "cpu":
        device["dtype"] = torch_module.float32
    else:
        device["dtype"] = torch_module.float16
    return device


# ================================================================================
# vram_usage_text
# GPU の使用量を返す。CUDA / ROCm 以外は None。
# allocated=実テンソル、reserved=キャッシュ込み、process=プロセス保持分。
# ================================================================================
def vram_usage_text(torch_module, device_index=0):
    cuda = getattr(torch_module, "cuda", None)

    if cuda is None or not cuda.is_available():
        return None

    try:
        free_bytes, total_bytes = cuda.mem_get_info(device_index)
    except (RuntimeError, TypeError, ValueError):
        return None

    used_gb = (total_bytes - free_bytes) / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    parts = [f"VRAM: {used_gb:.2f} / {total_gb:.2f} GB"]

    try:
        allocated_gb = cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved_gb = cuda.memory_reserved(device_index) / (1024 ** 3)
        parts.append(f"(allocated {allocated_gb:.2f} / reserved {reserved_gb:.2f})")
    except (RuntimeError, TypeError, ValueError):
        pass

    return " ".join(parts)


# ================================================================================
# reclaim_cuda_memory
# 未参照テンソルを回収し、CUDA キャッシュをドライバへ返す。
# 長時間 TTS での reserved VRAM の積み上がりを抑える。
# ================================================================================
def reclaim_cuda_memory(torch_module):
    import gc

    # 巨大チェックポイントは CPU キャッシュに残る。参照が消えたあとに返す。
    gc.collect()
    empty_host = getattr(getattr(torch_module, "_C", None), "_host_emptyCache", None)

    if empty_host is not None:
        try:
            empty_host()
        except RuntimeError:
            pass

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return

    try:
        cuda.empty_cache()
    except RuntimeError:
        pass


# ================================================================================
# _windows_ram_gb
# プロセスが保持している物理メモリと、システム全体の使用量を GB で返す。
# ================================================================================
def _windows_ram_gb():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None

    counters = ProcessMemory()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        return None

    gib = 1024 ** 3
    return {
        "process": counters.WorkingSetSize / gib,
        "system_used": (status.ullTotalPhys - status.ullAvailPhys) / gib,
        "system_total": status.ullTotalPhys / gib,
    }


# ================================================================================
# ram_usage_text
# メインメモリの使用量を返す。取れなければ None。
# ================================================================================
def ram_usage_text():
    if os.name != "nt":
        return None

    usage = _windows_ram_gb()

    if usage is None:
        return None

    return (
        "RAM: process {process:.2f} GB / "
        "system {system_used:.2f} / {system_total:.2f} GB"
    ).format(**usage)


# ================================================================================
# print_vram_usage
# ターミナルに VRAM とメインメモリの使用量を出す。
# ================================================================================
def print_vram_usage(torch_module, device_index=None):
    if device_index is None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            try:
                device_index = cuda.current_device()
            except RuntimeError:
                device_index = 0
        else:
            device_index = 0

    text = vram_usage_text(torch_module, device_index)

    if text:
        print(text)

    ram_text = ram_usage_text()

    if ram_text:
        print(ram_text)


# ================================================================================
# print_torch_device
# 起動ログに PyTorch とデバイス情報を出す。
# ================================================================================
def print_torch_device(torch_module, device):
    print("PyTorch:", torch_module.__version__)
    print("CUDA available:", torch_module.cuda.is_available())
    print("Device:", device["label"])
    device_map = str(device.get("device_map") or "")
    if device_map.startswith("cuda:"):
        print_vram_usage(torch_module, int(device_map.split(":", 1)[1]))


if __name__ == "__main__":
    require_nvidia_driver()

# ================================================================================
