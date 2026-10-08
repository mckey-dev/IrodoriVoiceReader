# ================================================================================
# miopen_log.py
#
# MIOpen の IsEnoughWorkspace 警告を 1 行にまとめる。
# Loading weights は tqdm のバーを、同じ 1 行の上で更新する。
# Windows では fd を差し替えず、Python の stderr だけを見る。
# ================================================================================

import atexit
import os
import re
import sys
import threading


_WARNING_MARK = "MIOpen: Warning [IsEnoughWorkspace]"
_WORKSPACE_RE = re.compile(r"workspace required:\s*(\d+)")
_ANSI_CURSOR = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_orig_stream = None
_filter_stream = None
_write_lock = threading.Lock()
_buf_lock = threading.Lock()
_state_lock = threading.Lock()
_flushing = False
_count = 0
_max_workspace = 0
_flush_timer = None
_installed = False
_stdout_filter_installed = False
_on_progress_line = False

_PROGRESS_PREFIXES = (
    "Loading weights:",
    "Downloading bytes:",
    "Download complete:",
    "Reconstructing",
    "Reconstruction complete:",
    "Fetching ",
)


# ================================================================================
# configure_stdio
# Windows のコンソールを UTF-8 にし、日本語の print が化けないようにする。
# ================================================================================
def configure_stdio():
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    encoding = "utf-8"

    for name in ("stdout", "stderr", "__stdout__", "__stderr__"):
        stream = getattr(sys, name, None)

        if stream is None or not hasattr(stream, "reconfigure"):
            continue

        try:
            stream.reconfigure(encoding=encoding, errors="replace")
        except Exception:
            pass

    return encoding


# ================================================================================
# _write_orig
# 元の stderr へ、そのまま書き出す。
# ================================================================================
def _write_orig(text):
    if not text:
        return

    stream = _orig_stream or sys.__stderr__

    with _write_lock:
        if stream is None:
            return

        try:
            stream.write(text)
            stream.flush()
        except Exception:
            pass


# ================================================================================
# _drain_filter_buffer
# 未完了の stderr バッファを確定する。
# ================================================================================
def _drain_filter_buffer():
    stream = _filter_stream

    if stream is None:
        return

    with _buf_lock:
        if not stream._buf:
            return

        leftover = stream._buf
        stream._buf = ""

    if leftover.strip():
        _handle_line(leftover)


# ================================================================================
# _record_warning
# MIOpen 警告なら溜めて True を返す。
# ================================================================================
def _record_warning(line):
    global _count, _max_workspace

    if "\r" in line:
        line = line.rsplit("\r", 1)[-1]

    if _WARNING_MARK not in line:
        return False

    match = _WORKSPACE_RE.search(line)

    with _state_lock:
        _count += 1

        if match:
            _max_workspace = max(_max_workspace, int(match.group(1)))

    return True


# ================================================================================
# _emit_pending
# 溜めた MIOpen 警告を 1 行出す。
# ================================================================================
def _emit_pending():
    global _count, _max_workspace, _flush_timer

    with _state_lock:
        if _flush_timer is not None:
            _flush_timer.cancel()
            _flush_timer = None

        count = _count
        max_workspace = _max_workspace
        _count = 0
        _max_workspace = 0

    if count <= 0:
        return

    if count == 1:
        _write_orig(
            f"{_WARNING_MARK} Solver <GemmFwdRest>, "
            f"workspace required: {max_workspace}\n"
        )
        return

    _write_orig(
        f"{_WARNING_MARK} Solver <GemmFwdRest> "
        f"x{count} (workspace max {max_workspace})\n"
    )


# ================================================================================
# flush_miopen_warnings
# 溜めた MIOpen 警告をまとめて出す。
# ================================================================================
def _end_progress_line():
    global _on_progress_line

    if not _on_progress_line:
        return

    _on_progress_line = False
    _write_orig("\n")


def flush_miopen_warnings():
    global _flushing

    if _flushing:
        return

    _flushing = True

    try:
        _drain_filter_buffer()
        _end_progress_line()
        _emit_pending()
    finally:
        _flushing = False


# ================================================================================
# _schedule_flush
# 警告の連打が止まったら、まとめて 1 行出す。
# ================================================================================
def _schedule_flush():
    global _flush_timer

    with _state_lock:
        if _flush_timer is not None:
            _flush_timer.cancel()

        _flush_timer = threading.Timer(0.3, flush_miopen_warnings)
        _flush_timer.daemon = True
        _flush_timer.start()


# ================================================================================
# _handle_line
# 確定した 1 行を処理する。
# ================================================================================
def _looks_like_progress(text):
    return text.startswith(_PROGRESS_PREFIXES)


def _is_finished_bar(text):
    return text.startswith(("Download complete:", "Reconstruction complete:"))


def _handle_line(line):
    global _on_progress_line

    if _record_warning(line):
        _schedule_flush()
        return

    if "\r" in line:
        line = line.rsplit("\r", 1)[-1]

    text = line.rstrip("\n")

    if "_grid_desc_" in text:
        return

    # カーソル移動だけの空行は捨て、バーの確定改行は 1 回だけ出す。
    if not text.strip():
        if _on_progress_line:
            _on_progress_line = False
            _write_orig("\n")
        return

    if _looks_like_progress(text):
        shown = text.rstrip()
        _on_progress_line = False
        _write_orig("\r" + shown + "\n")
        return

    if _on_progress_line:
        _on_progress_line = False
        _write_orig("\n")

    if not _flushing:
        flush_miopen_warnings()

    _write_orig(text + "\n")


# ================================================================================
# _FilterStderr
# tqdm の \r 更新はそのまま通し、改行済みの行だけを見る。
# isatty は True にして、tqdm が毎回改行しないようにする。
# ================================================================================
class _FilterStderr:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._buf = ""

    def write(self, text):
        global _on_progress_line

        if not text:
            return 0

        if not isinstance(text, str):
            text = str(text)

        written = len(text)
        text = _ANSI_CURSOR.sub("", text)

        lines = []
        progress = None

        with _buf_lock:
            self._buf += text

            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                lines.append(line)

            if "\r" in self._buf:
                prefix, current = self._buf.rsplit("\r", 1)
                # 表示済みの進捗は残さない。残すと確定行と連結して二重になる。
                self._buf = ""
                progress = (prefix, current)

        for line in lines:
            _handle_line(line + "\n")

        if progress is not None:
            prefix, current = progress

            if prefix.strip():
                _handle_line(prefix + "\n")

            if current and not _record_warning(current):
                if _is_finished_bar(current.lstrip()):
                    _on_progress_line = False
                    _write_orig("\r" + current.strip() + "\n")
                else:
                    _on_progress_line = True
                    _write_orig("\r" + current)

        return written

    def flush(self):
        try:
            self._wrapped.flush()
        except Exception:
            pass

    def isatty(self):
        return True

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# ================================================================================
# _install_ck_stdout_filter
# MIOpen の CK 畳み込みは、新しい形状になるとグリッド寸法を stdout へ直接書く。
# CK_LOGGING=0 でも残るので、その行だけ捨てる。
# ================================================================================
def _drop_ck_grid_line(line):
    return b"_grid_desc_" in line


def _forward_stdout_line(line, orig_fd):
    if _drop_ck_grid_line(line):
        return

    if line.endswith(b"\r"):
        os.write(orig_fd, line + b"\n")
        return

    os.write(orig_fd, line + b"\r\n")


def _read_filtered_stdout(read_fd, orig_fd):
    buf = b""

    try:
        while True:
            chunk = os.read(read_fd, 8192)

            if not chunk:
                break

            buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                _forward_stdout_line(line, orig_fd)

            if len(buf) > 65536 and b"\n" not in buf:
                if not _drop_ck_grid_line(buf):
                    os.write(orig_fd, buf)
                buf = b""
    except OSError:
        return

    if buf and not _drop_ck_grid_line(buf):
        os.write(orig_fd, buf)


def _reopen_stdout():
    # コンソール用の WriteConsole のままだと、パイプへ書いた瞬間に失敗する。
    old = sys.stdout

    try:
        old.flush()
    except Exception:
        pass

    try:
        old.detach()
    except Exception:
        pass

    sys.stdout = open(
        1,
        mode="w",
        encoding="utf-8",
        errors="replace",
        closefd=False,
        buffering=1,
    )
    sys.__stdout__ = sys.stdout


def _install_ck_stdout_filter():
    global _stdout_filter_installed

    if _stdout_filter_installed or sys.platform != "win32":
        return

    import msvcrt

    orig_fd = os.dup(1)
    read_fd, write_fd = os.pipe()
    reader = threading.Thread(
        target=_read_filtered_stdout,
        args=(read_fd, orig_fd),
        daemon=True,
    )
    reader.start()

    os.dup2(write_fd, 1)
    os.close(write_fd)
    msvcrt.setmode(1, os.O_BINARY)
    msvcrt.setmode(orig_fd, os.O_BINARY)

    import ctypes

    ctypes.windll.kernel32.SetStdHandle(
        -11,
        msvcrt.get_osfhandle(1),
    )
    _reopen_stdout()
    _stdout_filter_installed = True


# ================================================================================
# install_miopen_log_filter
# C++ の警告はログレベルで抑え、Python の stderr だけをまとめる。
# ================================================================================
def install_miopen_log_filter():
    global _orig_stream, _filter_stream, _installed

    if _installed:
        return

    configure_stdio()
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
    _install_ck_stdout_filter()

    _orig_stream = sys.stderr
    _filter_stream = _FilterStderr(_orig_stream)
    sys.stderr = _filter_stream
    _installed = True

    atexit.register(flush_miopen_warnings)

# ================================================================================
