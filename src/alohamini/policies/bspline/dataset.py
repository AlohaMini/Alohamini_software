"""Episode-bounded BSP labels backed by per-episode NumPy cache files."""

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from alohamini.datasets.statistics import RunningQuantileStats

from .codec import encode_sample, fit_episode


class BSplineDataset(torch.utils.data.Dataset):
    def __init__(self, source, options):
        self.source = source
        self.degree = options.get("degree", 3)
        self.options = {
            "degree": self.degree,
            "parameter_rows": options.get("horizon", 16),
            "max_error": options.get("fit_tolerance", 0.01),
            "stride": options.get("knot_stride", 1),
        }
        identity = dict(
            source=source.table_sha256,
            episodes=source.episodes,
            options=self.options,
            action=source.info["features"]["action"],
            version=1,
        )
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = (
            options.get("parameter_cache_dir") or Path(tempfile.gettempdir()) / "alohamini-bspline"
        )
        self.cache = Path(root) / digest
        self.cache.mkdir(parents=True, exist_ok=True)
        self._files = {}
        self._maps = OrderedDict()
        start = 0
        while start < len(source.rows):
            end = int(source.segment_ends[start])
            path = self.cache / f"episode_{start}_{end}.npy"
            if not path.exists():
                actions = np.asarray([source.rows[i]["action"] for i in range(start, end)])
                chunks = fit_episode(actions, **self.options)
                temporary = path.with_name(path.name + f".{uuid4().hex}.pending")
                try:
                    with temporary.open("xb") as stream:
                        np.save(stream, chunks, allow_pickle=False)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            self._files[start] = path
            start = end

    def __getattr__(self, name):
        source = self.__dict__.get("source")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    def __getstate__(self):
        return {**self.__dict__, "_maps": OrderedDict()}

    def __len__(self):
        return len(self.source)

    def parameters(self, index):
        row = self.sample_indices[index]
        start = int(self.segment_starts[row])
        chunks = self._maps.pop(start, None)
        if chunks is None:
            chunks = np.load(self._files[start], mmap_mode="r", allow_pickle=False)
            if chunks.ndim != 3 or chunks.shape[1] != self.options["parameter_rows"]:
                raise ValueError("Invalid cached BSP parameter layout")
        self._maps[start] = chunks
        while len(self._maps) > 4:
            self._maps.popitem(last=False)
        return encode_sample(chunks, row - start, self.degree)

    def __getitem__(self, index):
        return {**self.source[index], "trajectory": torch.from_numpy(self.parameters(index))}

    def trajectory_statistics(self):
        tracker = RunningQuantileStats()
        for i in range(len(self)):
            tracker.update(self.parameters(i))
        return {name: value.tolist() for name, value in tracker.get_statistics().items()}
