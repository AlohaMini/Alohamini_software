"""Run native simulation viewers in an owned, software-presented X11 display."""

import os
import secrets
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def isolated_window(mode, *, environment=None, drm_directory=Path("/sys/class/drm")):
    environment = os.environ if environment is None else environment
    if mode not in ("auto", "native", "isolated"):
        raise ValueError("Window mode must be auto, native or isolated")
    if mode != "auto":
        return mode == "isolated"
    if sys.platform != "linux" or environment.get("XDG_SESSION_TYPE", "x11") != "x11":
        return False
    if not environment.get("DISPLAY"):
        return False
    vendors = {
        path.read_text().strip().lower() for path in drm_directory.glob("card*/device/vendor")
    }
    return "0x10de" in vendors and len(vendors) > 1


def _stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run_isolated_window(command, *, title):
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        raise ValueError("An isolated viewer requires a Linux X11 desktop (DISPLAY)")
    xephyr, xauth = shutil.which("Xephyr"), shutil.which("xauth")
    if xephyr is None or xauth is None:
        raise FileNotFoundError("Install xserver-xephyr and xauth; see docs/install.md")
    workspace = Path(os.environ.get("ALOHAMINI_WORKSPACE", "~/Alohamini_workspace")).expanduser()
    log_path = workspace / "simulation/viewer" / f"x11-{time.time_ns()}-{os.getpid()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server = child = None
    with tempfile.TemporaryDirectory(prefix="alohamini-x11-") as directory:
        authority = Path(directory) / "Xauthority"
        authority.touch(mode=0o600)
        cookie = secrets.token_hex(16)

        def authorize(display):
            subprocess.run(
                [xauth, "-f", str(authority), "add", display, "MIT-MAGIC-COOKIE-1", cookie],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        authorize(":0")
        read_fd, write_fd = os.pipe()
        try:
            with log_path.open("w") as log:
                try:
                    server = subprocess.Popen(
                        [
                            xephyr,
                            "-displayfd",
                            str(write_fd),
                            "-screen",
                            "1920x1080",
                            "-resizeable",
                            "-noreset",
                            "-no-host-grab",
                            "-nolisten",
                            "tcp",
                            "-auth",
                            str(authority),
                            "-title",
                            title,
                        ],
                        pass_fds=(write_fd,),
                        stdout=subprocess.DEVNULL,
                        stderr=log,
                    )
                finally:
                    os.close(write_fd)
                if not select.select([read_fd], [], [], 10)[0]:
                    raise RuntimeError(f"Isolated display startup timed out; see {log_path}")
                number = os.read(read_fd, 64).decode().strip()
                if not number.isdecimal() or server.poll() is not None:
                    raise RuntimeError(f"Cannot start the isolated display; see {log_path}")
                display = f":{number}"
                authorize(display)
                environment = {**os.environ, "DISPLAY": display, "XAUTHORITY": str(authority)}
                print("Viewer: isolated X11 window", flush=True)
                child = subprocess.Popen(command, env=environment)
                while child.poll() is None:
                    if server.poll() is not None:
                        _stop(child)
                        return 0
                    time.sleep(0.05)
                return child.returncode
        finally:
            os.close(read_fd)
            try:
                _stop(child)
            finally:
                _stop(server)
