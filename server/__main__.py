# ================================================================================
# python -m server
#
# ローカル TTS サーバーを起動する。
# HOST / PORT は既定のまま、起動引数 --host / --port で上書きできる。
# ================================================================================

import argparse
import os
import socket
import sys

import uvicorn

from server.config import HOST, PORT
from server.gpu_runtime import (
    DEVICE_ENV,
    configure_backend_env,
    default_device_id,
    driver_too_old_message,
    format_device_line,
    normalize_device_choice,
    print_skipped_devices,
    query_compute_devices,
    use_integrated_gpu_alone,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Irodori-TTS ローカル TTS サーバーを起動する。",
    )
    parser.add_argument(
        "--host",
        default=HOST,
        help=f"待ち受けホスト（既定: {HOST}）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"待ち受けポート（既定: {PORT}）",
    )
    parser.add_argument(
        "--device",
        default="",
        help="使うデバイス。cpu または cuda:0。省略時は起動時に番号で選ぶ。",
    )
    return parser.parse_args()


# ================================================================================
# prompt_device
# 番号でデバイスを選ばせる。Enter は既定（単体 GPU、無ければ CPU）。
# ================================================================================
def prompt_device(devices):
    chosen_id = default_device_id(devices)
    default_number = next(
        number
        for number, device in enumerate(devices, start=1)
        if device["id"] == chosen_id
    )

    print("使用するデバイスを選んでください。")
    for number, device in enumerate(devices, start=1):
        suffix = "  ← 既定" if number == default_number else ""
        print(f"  {number}) {format_device_line(device)}{suffix}")

    while True:
        try:
            answer = input(f"番号 [{default_number}]: ").strip()
        except EOFError:
            answer = ""
        except KeyboardInterrupt:
            print()
            raise SystemExit(1)

        if answer == "":
            return chosen_id

        if answer.isdigit():
            number = int(answer)
            if 1 <= number <= len(devices):
                return devices[number - 1]["id"]

        print("一覧の番号を入力してください。")


# ================================================================================
# prepare_device
# 起動前にデバイスを決め、IRODORI_DEVICE に書く。
# ================================================================================
def prepare_device(requested):
    print("デバイスを確認しています...")
    configure_backend_env()
    devices, gpu_build = query_compute_devices()

    if requested:
        choice = normalize_device_choice(requested, devices)
    else:
        detected_gpu = any(device["id"] != "cpu" for device in devices)
        print_skipped_devices(devices)
        devices = [device for device in devices if device.get("supported", True)]
        gpu_devices = [device for device in devices if device["id"] != "cpu"]

        if gpu_devices and sys.stdin.isatty():
            choice = prompt_device(devices)
        elif gpu_devices:
            choice = default_device_id(devices)
        elif gpu_build and not detected_gpu:
            print()
            print(driver_too_old_message())
            raise SystemExit(1)
        else:
            choice = "cpu"

    return use_integrated_gpu_alone(choice, devices)


# ================================================================================
# ensure_port_available
# 待ち受けポートが空いていることを確認する。
# ================================================================================
def ensure_port_available(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.bind((host, port))
    except OSError:
        print()
        print(f"{port} 番ポートは既に使われています。")
        print("前のサーバーを閉じてから、もう一度起動してください。")
        print(f"例: netstat -ano | findstr :{port}")
        raise SystemExit(1)
    finally:
        sock.close()


if __name__ == "__main__":
    args = parse_args()
    host = args.host
    port = args.port
    ensure_port_available(host, port)
    os.environ[DEVICE_ENV] = prepare_device(args.device)

    print()
    print("=" * 60)
    print(f"Server starting: http://{host}:{port}")
    print("=" * 60)
    print()

    from server.app import app

    uvicorn.run(
        app,
        host=host,
        port=port,
    )

# ================================================================================
