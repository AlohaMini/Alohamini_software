"""Disk-backed recording encoders isolated from the capture process.

Only paths cross the process boundary. The accepted-frame journal remains the
authority, including after image rejection, early stop and an interrupted save.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from alohamini.datasets.encoding_process import resident_bytes, watch_parent


def validate_video_options(encoder, background, memory_mib):
    if encoder not in ("cpu", "nvenc", "videotoolbox"):
        raise ValueError("Video encoder must be cpu, nvenc or videotoolbox")
    if type(background) is not bool:
        raise ValueError("background_video_encoding must be a bool")
    if type(memory_mib) is not int or memory_mib < 256:
        raise ValueError("Video encoding memory must be an integer of at least 256 MiB")


def encoder_settings(encoder):
    if encoder == "cpu":
        return "libx264", {"crf": "18", "preset": "fast", "g": "2"}
    if encoder == "nvenc":
        # QP and CRF are different controls; this does not claim pixel equivalence.
        return "h264_nvenc", {
            "preset": "p4",
            "rc": "constqp",
            "qp": "18",
            "g": "2",
            "bf": "0",
            "gpu": "0",
        }
    if encoder == "videotoolbox":
        return "h264_videotoolbox", {
            "b": "8000000", "g": "2", "bf": "0", "allow_sw": "0", "realtime": "1",
        }
    raise ValueError("Video encoder must be cpu, nvenc or videotoolbox")


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
    temporary.replace(path)


class RecordingVideoEncoder:
    """One bounded encoder process per episode, with no capture-side image queue."""

    def __init__(self, episode, fps, cameras, *, encoder, workers, memory_mib, profile):
        self.episode = Path(episode)
        self.directory = self.episode / ".video"
        self.directory.mkdir()
        self.process = None
        self._stop = threading.Event()
        self._monitor = None
        self._parent_pipe = None
        self._reason = None
        self._warned = False
        self._memory_bytes = memory_mib * 1024**2
        self._fps = fps
        self._cameras = tuple(cameras)
        config = dict(
            fps=fps,
            cameras=list(cameras),
            encoder=encoder,
            workers=workers,
            memory_mib=memory_mib,
            profile=profile,
            parent_pid=os.getpid(),
        )
        env = os.environ.copy()
        for key in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = "1"
        # Also work from an uninstalled checkout and under tests using PYTHONPATH.
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
        parent_read = None
        try:
            parent_read, self._parent_pipe = os.pipe()
            config["parent_fd"] = parent_read
            _write(self.directory / "config.json", config)
            with (self.directory / "worker.log").open("wb") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", __name__, str(self.directory / "config.json")],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    # Terminal Ctrl+C belongs to the recorder. Its cleanup
                    # publishes finished.json so the encoder can drain normally.
                    start_new_session=True,
                    pass_fds=(parent_read,),
                )
            self._monitor = threading.Thread(
                target=self._watch, name="AlohaMiniVideoMemory", daemon=True
            )
            self._monitor.start()
        except (OSError, RuntimeError) as exc:
            self._reason = f"could not start encoder: {exc}"
            self.cancel()
            self._warning()
        finally:
            if parent_read is not None:
                os.close(parent_read)

    def _warning(self):
        if not self._warned:
            logging.warning(
                "Background video encoding unavailable (%s); original JPEGs retained; "
                "CPU H.264 encoding will run at save.",
                self._reason,
            )
            self._warned = True

    def _watch(self):
        while not self._stop.wait(0.1):
            if self.process.poll() is not None:
                return
            try:
                rss = resident_bytes(self.process.pid)
                if rss > self._memory_bytes:
                    self._reason = "encoder exceeded its RAM budget"
                    self.process.kill()
                    self._warning()
                    return
            except (OSError, ValueError) as exc:
                if self.process.poll() is not None:
                    return
                self._reason = f"encoder memory monitoring unavailable: {exc}"
                self.process.kill()
                self._warning()
                return

    def finish(self, length, references):
        """Drain disk backlog after capture, returning only a complete matching result."""
        if self.process is None:
            return None
        try:
            _write(self.directory / "finished.json", {"length": length})
            self.process.wait(timeout=max(30.0, 3 * length / self._fps))
            if self.process.returncode:
                raise RuntimeError(f"encoder exited with status {self.process.returncode}")
            result = json.loads((self.directory / "result.json").read_text())
            if result.get("error"):
                raise RuntimeError(result["error"])
            if set(result["cameras"]) != set(self._cameras):
                raise ValueError("encoder camera set changed")
            for camera, detail in result["cameras"].items():
                if detail["frames"] != length or detail["references_sha256"] != references[camera]:
                    raise ValueError(f"encoder input order/count differs: {camera}")
            if result.get("warning"):
                logging.warning("Video encoder: %s", result["warning"])
            return result
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self._reason = self._reason or str(exc)
            # A worker may have managed to report its more useful native error.
            try:
                self._reason = json.loads((self.directory / "result.json").read_text()).get(
                    "error", self._reason
                )
            except (OSError, ValueError):
                pass
            self._warning()
            return None
        finally:
            self.cancel()

    def cancel(self):
        """Stop readers/writers before deleting or moving the pending directory."""
        self._stop.set()
        if self._parent_pipe is not None:
            os.close(self._parent_pipe)
            self._parent_pipe = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._monitor is not None and self._monitor.ident is not None:
            self._monitor.join(timeout=1)


def _references(episode, camera):
    """Tail complete accepted rows; backlog stays in the journal and JPEG files."""
    journal = episode / "journal.jsonl"
    finished = episode / ".video/finished.json"
    while not journal.exists():
        if finished.exists():
            raise ValueError("Recording journal is missing")
        time.sleep(0.01)
    count, partial = 0, ""
    with journal.open(encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if line:
                partial += line
                if not partial.endswith("\n"):
                    continue
                item = json.loads(partial)
                partial = ""
                if "frame" not in item:
                    continue
                row = item["frame"]
                if row["frame_index"] != count:
                    raise ValueError("Non-contiguous recording frame order")
                yield row[f"observation.images.{camera}"]
                count += 1
            elif finished.exists():
                # The writer could have appended between our EOF read and
                # observing finished.json. Recheck after that publication.
                position = stream.tell()
                if stream.readline():
                    stream.seek(position)
                    continue
                if partial or count != json.loads(finished.read_text())["length"]:
                    raise ValueError("Incomplete recording journal")
                return
            else:
                time.sleep(0.01)


def _encode_camera(episode, camera, config, encoder):
    import itertools

    import av

    from alohamini.datasets.images import image_rgb
    from alohamini.datasets.video import _ENCODER_THREADS, _encode_frames, inspect_video

    started = time.perf_counter()
    references = iter(_references(episode, camera))
    first = next(references)
    first_rgb = image_rgb(episode, camera, first)
    shape = list(first_rgb.shape)
    digest = hashlib.sha256()
    detail = {} if config["profile"] else None

    def frames():
        for index, reference in enumerate(itertools.chain((first,), references)):
            digest.update((reference + "\n").encode())
            rgb = first_rgb if index == 0 else image_rgb(episode, camera, reference)
            if list(rgb.shape) != shape:
                raise ValueError(f"Camera resolution changed: {camera}")
            yield av.VideoFrame.from_ndarray(rgb, format="rgb24")

    codec, options = encoder_settings(encoder)
    pix_fmt = "yuv444p" if shape[0] % 2 or shape[1] % 2 else "yuv420p"
    path = episode / "videos" / f"{camera}.mp4"
    count = _encode_frames(
        frames(),
        path,
        config["fps"],
        shape,
        codec=codec,
        pix_fmt=pix_fmt,
        options=options,
        timings=detail,
    )
    if inspect_video(path) != {"frames": count, "fps": config["fps"], "shape": shape}:
        raise ValueError(f"Encoded video headers differ: {camera}")
    if detail is not None:
        detail.update(
            codec=codec,
            pix_fmt=pix_fmt,
            frames=count,
            width=shape[1],
            height=shape[0],
            threads=_ENCODER_THREADS,
            gop=2,
            wall_s=time.perf_counter() - started,
        )
        if encoder == "videotoolbox":
            detail["bit_rate"] = int(options["b"])
        else:
            detail["preset"] = options["preset"]
            detail["crf" if encoder == "cpu" else "qp"] = 18
    return camera, dict(
        frames=count, shape=shape, references_sha256=digest.hexdigest(), timings=detail
    )


def _worker(config_path):
    """Entry point deliberately imports numerical/native libraries after limits."""
    import ctypes
    import resource
    import signal
    from concurrent.futures import ThreadPoolExecutor

    config_path = Path(config_path)
    directory, episode = config_path.parent, config_path.parent.parent
    try:
        config = json.loads(config_path.read_text())
        watch_parent(config["parent_fd"], config["parent_pid"])
        if sys.platform.startswith("linux"):
            # Do not leave an orphan encoder writing after a crashed recorder.
            if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0):
                raise OSError(ctypes.get_errno(), "Could not set encoder parent-death signal")
            if os.getppid() != config["parent_pid"]:
                return 1
        budget = config["memory_mib"] * 1024**2
        # DATA bounds writable allocations without blocking CUDA's large virtual
        # address reservations. RLIMIT_RSS is not enforced on modern Linux.
        if sys.platform.startswith("linux"):
            _, hard = resource.getrlimit(resource.RLIMIT_DATA)
            budget = budget if hard == resource.RLIM_INFINITY else min(budget, hard)
            resource.setrlimit(resource.RLIMIT_DATA, (budget, budget))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.nice(5)
        from alohamini.datasets.video import _encode_frames

        encoder, warning = config["encoder"], None
        if encoder != "cpu":
            import av
            import numpy as np

            codec, options = encoder_settings(encoder)
            probe = directory / "probe.mp4"
            try:
                _encode_frames(
                    [
                        av.VideoFrame.from_ndarray(
                            np.zeros((480, 640, 3), dtype=np.uint8), format="rgb24"
                        )
                    ],
                    probe,
                    config["fps"],
                    [480, 640, 3],
                    codec=codec,
                    options=options,
                )
            except Exception as exc:
                encoder = "cpu"
                label = "NVENC" if config["encoder"] == "nvenc" else "VideoToolbox"
                warning = f"{label} unavailable ({type(exc).__name__}: {exc}); using CPU H.264."
            finally:
                probe.unlink(missing_ok=True)
        (episode / "videos").mkdir(exist_ok=True)
        cameras = config["cameras"]
        workers = min(config["workers"] or len(cameras), len(cameras))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="AlohaMiniVideo") as pool:
            results = dict(pool.map(lambda c: _encode_camera(episode, c, config, encoder), cameras))
        _write(
            directory / "result.json", {"cameras": results, "encoder": encoder, "warning": warning}
        )
        return 0
    except Exception as exc:
        _write(directory / "result.json", {"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(_worker(sys.argv[1]))
