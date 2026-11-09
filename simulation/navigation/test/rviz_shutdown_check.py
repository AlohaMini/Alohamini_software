"""Check real RViz shutdown without hardware or Nav2 servers (requires Xvfb and python3-xlib)."""

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from Xlib import X, display, protocol


def close_window(pid):
    connection = display.Display()
    try:
        atom = connection.intern_atom("_NET_WM_PID")
        delete = connection.intern_atom("WM_DELETE_WINDOW")
        windows = list(connection.screen().root.query_tree().children)
        for window in windows:
            value = window.get_full_property(atom, X.AnyPropertyType)
            if (
                value is not None
                and int(value.value[0]) == pid
                and window.get_attributes().map_state == X.IsViewable
                and delete in (window.get_wm_protocols() or [])
            ):
                window.send_event(
                    protocol.event.ClientMessage(
                        window=window,
                        client_type=connection.intern_atom("WM_PROTOCOLS"),
                        data=(32, [delete, X.CurrentTime, 0, 0, 0]),
                    ),
                    event_mask=0,
                )
                connection.sync()
                return
        raise AssertionError(f"RViz window not found for PID {pid}")
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = Path(get_package_prefix("rviz2")) / "lib/rviz2/rviz2"
    plugin = Path(get_package_prefix("nav2_rviz_plugins")) / "lib/libnav2_rviz_plugins.so"
    image_plugin = Path(get_package_prefix("alohamini_rviz")) / "lib/libalohamini_rviz.so"
    config = Path(get_package_share_directory("alohamini_navigation")) / "config/navigation.rviz"
    assert not str(plugin).startswith("/opt/ros/"), (
        "Source the workspace containing the patched plugin"
    )
    results = []
    for method in ("SIGINT", "SIGTERM", "window_close"):
        for iteration in range(args.repetitions):
            log = output / f"{method}-{iteration}.log"
            with log.open("w") as stream:
                process = subprocess.Popen(
                    [str(executable), "-d", str(config)],
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        assert process.poll() is None, log.read_text()
                        loaded = Path(f"/proc/{process.pid}/maps").read_text()
                        if str(plugin) in loaded and str(image_plugin) in loaded:
                            break
                        time.sleep(0.1)
                    else:
                        raise AssertionError("RViz did not load the patched plugin")
                    time.sleep(2)
                    start = time.monotonic()
                    if method == "window_close":
                        close_window(process.pid)
                    else:
                        process.send_signal(getattr(signal, method))
                    code = process.wait(timeout=5)
                    results.append(
                        {
                            "method": method,
                            "iteration": iteration,
                            "returncode": code,
                            "seconds": time.monotonic() - start,
                        }
                    )
                    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
                    assert code == 0, log.read_text()
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
