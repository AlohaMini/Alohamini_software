"""Immutable prepared tensor features with atomic publication and bounded CPU reuse."""

import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def source_signature(paths):
    """Track local immutable encoder assets without rereading all weight contents."""
    result = []
    for source in paths:
        source = Path(source).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            if any(part.startswith(".") for part in path.relative_to(source).parts):
                continue
            stat = path.stat()
            result.append(
                (
                    str(source),
                    str(path.relative_to(source)),
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )
    return result


def _load_manifest(path):
    path = Path(path)
    report = json.loads((path / "features.json").read_text())
    if report.get("format") != "alohamini-frozen-features" or report.get("version") != 1:
        raise ValueError("Unsupported frozen feature store")
    entries = report.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Frozen feature store requires named entries")
    files = set()
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError("Invalid frozen feature entry")
        name = entry.get("file", "")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            or name in files
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or not isinstance(entry.get("shape"), list)
        ):
            raise ValueError("Invalid frozen feature index")
        files.add(name)
        target = path / name
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != entry.get("bytes")
        ):
            raise ValueError(f"Missing or modified frozen feature file: {name}")
    return report


def prepare_features(root, signature, keys, producer, *, batch_size=8):
    """Build features on the caller's designated writer rank; never publish partial output.

    Keys are exact canonical strings. ``producer`` returns one immutable tensor
    for each requested key; model-specific tokenization stays outside this module.
    """
    import fcntl

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Frozen feature batch size must be a positive integer")
    if (
        not isinstance(keys, (list, tuple))
        or not keys
        or any(not isinstance(k, str) or not k for k in keys)
    ):
        raise ValueError("Frozen feature keys must be nonempty strings")
    keys = sorted(set(keys))
    encoded = json.dumps(
        dict(signature=signature, keys=keys, format=1), sort_keys=True, allow_nan=False
    )
    identity = json.loads(encoded)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / digest
    with (root / (digest + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            report = _load_manifest(path)
            if report["identity"] != identity:
                raise ValueError("Frozen feature cache identity differs")
            return path
        temporary = Path(tempfile.mkdtemp(prefix="." + digest + "-", dir=root))
        try:
            entries = {}
            for start in range(0, len(keys), batch_size):
                selected = keys[start : start + batch_size]
                values = list(producer(selected))
                if len(values) != len(selected):
                    raise ValueError("Frozen feature producer returned a different batch size")
                for key, value in zip(selected, values, strict=True):
                    if (
                        not isinstance(value, torch.Tensor)
                        or not value.is_floating_point()
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError("Frozen feature producer requires finite floating tensors")
                    if value.requires_grad:
                        raise ValueError(
                            "Trainable activations cannot enter a frozen feature cache"
                        )
                    name = hashlib.sha256(key.encode()).hexdigest() + ".safetensors"
                    target = temporary / name
                    value = value.detach().to("cpu").contiguous().clone()
                    save_file({"feature": value}, target)
                    entries[key] = dict(
                        file=name,
                        bytes=target.stat().st_size,
                        sha256=_digest_file(target),
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                    )
            report = dict(
                format="alohamini-frozen-features", version=1, identity=identity, entries=entries
            )
            (temporary / "features.json").write_text(
                json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return path


class FrozenFeatureCache:
    """A read-only disk store plus an LRU bounded by tensor bytes, independent of policy type."""

    def __init__(self, path=None, *, max_bytes=256 * 1024 * 1024):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Frozen feature byte budget must be a nonnegative integer")
        self.path = Path(path).resolve() if path else None
        self.manifest = _load_manifest(self.path) if self.path else None
        self.max_bytes = max_bytes
        self._values = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != "_lock"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = RLock()

    def _remember(self, key, value):
        value = value.detach().to("cpu").contiguous().clone()
        size = value.numel() * value.element_size()
        if size <= self.max_bytes:
            while self._values and self._bytes + size > self.max_bytes:
                _, old = self._values.popitem(last=False)
                self._bytes -= old.numel() * old.element_size()
            self._values[key] = value
            self._bytes += size
        return value

    def get_many(self, keys, producer):
        with self._lock:
            return self._get_many(keys, producer)

    def _get_many(self, keys, producer):
        result = {}
        missing = []
        for key in dict.fromkeys(keys):
            if key in self._values:
                self._values.move_to_end(key)
                result[key] = self._values[key]
                continue
            entry = (self.manifest or {}).get("entries", {}).get(key)
            if entry is None:
                missing.append(key)
                continue
            path = self.path / entry["file"]
            if _digest_file(path) != entry["sha256"]:
                raise ValueError(f"Frozen feature checksum differs: {entry['file']}")
            with safe_open(path, framework="pt", device="cpu") as stream:
                if set(stream.keys()) != {"feature"}:
                    raise ValueError("Unexpected frozen feature tensor structure")
                value = stream.get_tensor("feature")
            if list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]:
                raise ValueError("Frozen feature shape/dtype differs from its index")
            result[key] = self._remember(key, value)
        if missing:
            values = list(producer(missing))
            if len(values) != len(missing):
                raise ValueError("Frozen feature producer returned a different batch size")
            for key, value in zip(missing, values, strict=True):
                if (
                    not isinstance(value, torch.Tensor)
                    or not value.is_floating_point()
                    or value.requires_grad
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("Frozen feature producer requires detached finite tensors")
                result[key] = self._remember(key, value)
        # Callers may modify returned features without changing future predictions.
        return [result[key].clone() for key in keys]

    def copy_to(self, path):
        if self.path is None:
            raise ValueError("Only prepared frozen feature stores can be bundled")
        _load_manifest(self.path)
        shutil.copytree(self.path, path)
