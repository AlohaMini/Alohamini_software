"""Pass compressed media to the viewer; decode only explicitly requested keyframes."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from uuid import uuid4

from .source import IMAGE_PREFIX, Media, file_signature


class MediaReader:
    def __init__(self, cache: Path):
        self.cache = cache
        self._group_key = self._group = None
        self._video = None
        self.keyframe_requests = self.cache_hits = self.video_frame_requests = 0

    def close(self):
        if self._video is not None:
            self._video.close()
        self._group = None
        # Bound the persistent keyframe cache to 1 GiB, newest/recently used first.
        # Only files owned by this cache are considered.
        if self.cache.is_dir():
            files = sorted(
                self.cache.glob("[0-9a-f]" * 64 + ".png"),
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )
            retained = 0
            for path in files:
                retained += path.stat().st_size
                if retained > 1024**3:
                    path.unlink(missing_ok=True)

    def encoded(self, ref: Media, camera: str):
        if ref.kind in ("native_image", "native_host_image"):
            from alohamini.datasets.images import image_bytes

            return image_bytes(ref.path, camera, ref.reference)
        if ref.kind != "v3_image":
            raise ValueError("Video must use a compressed asset or an explicit keyframe request")
        value = self._cell(ref, camera)
        if value.get("bytes"):
            return value["bytes"]
        return self._external_path(ref, value).read_bytes()

    def _cell(self, ref, camera):
        import pyarrow.parquet as pq

        key = (ref.path, ref.group, camera)
        if self._group_key != key:
            with pq.ParquetFile(ref.path) as file:
                self._group = file.read_row_group(ref.group, columns=[IMAGE_PREFIX + camera])
            self._group_key = key
        value = self._group[IMAGE_PREFIX + camera][ref.index].as_py()
        if value is None:
            raise ValueError("嵌入图像缺失")
        return value

    @staticmethod
    def _external_path(ref, value):
        # Relative paths follow the dataset root, not the shard's directory.
        root = ref.path
        while root.parent != root and not (root / "meta/info.json").is_file():
            root = root.parent
        path = (root / value["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image reference escapes dataset")
        return path

    def keyframe(self, ref: Media, camera: str):
        from PIL import Image

        self.keyframe_requests += 1
        source_path = ref.path
        if ref.kind in ("native_image", "native_host_image"):
            from alohamini.datasets.images import image_path

            source_path = image_path(ref.path, camera, ref.reference)
        elif ref.kind == "v3_image":
            value = self._cell(ref, camera)
            if not value.get("bytes"):
                source_path = self._external_path(ref, value)
        signature = [
            str(source_path),
            file_signature(source_path),
            ref.kind,
            ref.index,
            ref.group,
            ref.reference,
            camera,
            "png-rgb-v1",
        ]
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.png"
        if path.is_file():
            self.cache_hits += 1
            path.touch()
            return path
        if ref.kind == "video":
            from alohamini.datasets.video_reader import VideoFrameCache

            if self._video is None:
                self._video = VideoFrameCache(max_size=4, backend="pyav")
            image = Image.fromarray(self._video.read(ref.path, ref.index))
            self.video_frame_requests += 1
        else:
            image = Image.open(io.BytesIO(self.encoded(ref, camera))).convert("RGB")
            if ref.kind == "native_host_image":
                # Historical Host JPEGs use OpenCV's BGR decoder output as RGB.
                # Standard image viewers need this channel correction exactly once.
                red, green, blue = image.split()
                image = Image.merge("RGB", (blue, green, red))
        self.cache.mkdir(parents=True, exist_ok=True)
        # PNG is an explicit keyframe export, never a video re-encode.
        temporary = self.cache / f".{key}.{uuid4().hex}.tmp"
        try:
            image.save(temporary, format="PNG")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
