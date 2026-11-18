"""Native process accounting and parent lifetime for isolated video encoders."""

import ctypes
import errno
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path


class _TaskInfo(ctypes.Structure):
    # Darwin proc_taskinfo ABI (bsd/sys/proc_info.h), identical on arm64 and x86_64.
    _fields_ = [
        ("virtual_bytes", ctypes.c_uint64),
        ("resident_bytes", ctypes.c_uint64),
        ("_times", ctypes.c_uint64 * 4),
        ("_counters", ctypes.c_int32 * 12),
    ]


@lru_cache(maxsize=1)
def _libproc():
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pidinfo.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
    ]
    library.proc_pidinfo.restype = ctypes.c_int
    return library


def resident_bytes(pid):
    """Current resident bytes, rather than ru_maxrss's historical high-water mark."""
    if sys.platform == "darwin":
        info = _TaskInfo()
        size = ctypes.sizeof(info)
        read = _libproc().proc_pidinfo(pid, 4, 0, ctypes.byref(info), size)
        if read != size:
            error = ctypes.get_errno() if read <= 0 else errno.EPROTO
            raise OSError(error or errno.ESRCH, "Could not read encoder resident memory")
        return info.resident_bytes
    if sys.platform.startswith("linux"):
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise OSError(errno.ESRCH, "Encoder resident memory is unavailable")
    raise OSError(errno.ENOSYS, "Encoder accounting supports Linux and macOS")


def watch_parent(fd, parent_pid):
    """Exit on pipe EOF even when a recorder is killed without running cleanup.

    Only the recorder owns the write end. pass_fds transfers the read end to the
    encoder; an OS-closed write end remains reliable across PID reuse.
    """
    if os.getppid() != parent_pid:
        raise RuntimeError("Recording parent exited before encoder startup")

    def watch():
        try:
            while os.read(fd, 1):
                pass
        finally:
            os.close(fd)
            # A daemon cannot wait for journal readers or a native codec to finish.
            os._exit(1)

    thread = threading.Thread(target=watch, name="AlohaMiniEncoderParent", daemon=True)
    thread.start()
    return thread
