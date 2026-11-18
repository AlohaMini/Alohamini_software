"""Strict local safetensors loading, including HF shards and tied parameters."""

import json
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def _files(path, filename="model.safetensors"):
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".safetensors")
    ):
        raise ValueError("Weights filename must be a local safetensors basename")
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError("Weights must be safetensors or a local checkpoint directory")
        return [(path, None)]
    single = path / filename
    index = path / (filename + ".index.json")
    if single.is_file() and index.is_file():
        raise ValueError("Checkpoint contains both single-file and sharded weights")
    if single.is_file():
        return [(single, None)]
    if not index.is_file():
        raise FileNotFoundError(single)
    mapping = json.loads(index.read_text()).get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Safetensors index requires a nonempty weight_map")
    groups = defaultdict(set)
    for key, filename in mapping.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError("Invalid safetensors shard name or tensor key")
        shard = path / filename
        # HF local snapshots intentionally link their shard names to blobs.
        if not shard.is_file():
            raise ValueError(f"Missing checkpoint shard: {filename}")
        groups[shard].add(key)
    return sorted(groups.items())


def load_weights(
    model,
    path,
    *,
    reinitialize=(),
    prefix="",
    key_mapping=None,
    exclude=(),
    filename="model.safetensors",
):
    """Validate the entire schema before loading one shard at a time.

    Only explicitly named prefixes may keep freshly initialized parameters.
    Shapes and shared-storage aliases are checked before any weight is changed.
    Explicit exclusions may remove source modules absent from the target model.
    Every shard's index and headers are still inspected before copying selected tensors.
    Returns those reinitialized names for logging and checkpoint provenance.
    """
    import torch

    if not isinstance(reinitialize, (tuple, list)) or any(
        not isinstance(prefix, str) or not prefix for prefix in reinitialize
    ):
        raise ValueError("reinitialize must contain explicit nonempty parameter prefixes")
    if not isinstance(prefix, str):
        raise ValueError("Checkpoint prefix must be a string")
    if not isinstance(exclude, (tuple, list)) or any(
        not isinstance(name, str) or not name for name in exclude
    ):
        raise ValueError("exclude requires explicit nonempty source module prefixes")
    key_mapping = {} if key_mapping is None else key_mapping
    if not isinstance(key_mapping, dict) or any(
        not isinstance(source, str) or not source or not isinstance(target, str) or not target
        for source, target in key_mapping.items()
    ):
        raise ValueError("Checkpoint key_mapping requires nonempty source and target names")

    def local_name(key):
        if not key.startswith(prefix) or key == prefix:
            raise ValueError(f"Checkpoint key does not match explicit prefix {prefix!r}: {key}")
        name = key.removeprefix(prefix)
        return key_mapping.get(name, name)

    files = _files(path, filename)
    own = model.state_dict()
    if any(name.startswith(tuple(exclude)) for name in own):
        raise ValueError("Excluded source modules must be absent from the target model")
    shapes, aliases = {}, {}
    for shard, expected in files:
        with safe_open(shard, framework="pt", device="cpu") as stream:
            keys = set(stream.keys())
            if expected is not None and keys != expected:
                raise ValueError(f"Safetensors shard disagrees with its index: {shard.name}")
            local_keys = {local_name(key) for key in keys}
            if len(local_keys) != len(keys):
                raise ValueError("Checkpoint key mapping has colliding target names")
            if local_keys & shapes.keys():
                raise ValueError("Duplicate tensors across checkpoint shards")
            shapes.update(
                {local_name(key): tuple(stream.get_slice(key).get_shape()) for key in keys}
            )
            aliases.update(
                {
                    local_name(key): local_name(value)
                    for key, value in (stream.metadata() or {}).items()
                    if key.startswith(prefix) and local_name(key) in own and value in keys
                }
            )
    shapes = {name: shape for name, shape in shapes.items() if not name.startswith(tuple(exclude))}
    unexpected = shapes.keys() - own.keys()
    if unexpected:
        raise ValueError(f"Unexpected checkpoint weights: {sorted(unexpected)}")
    mismatched = {key for key, shape in shapes.items() if shape != tuple(own[key].shape)}
    shared = defaultdict(set)
    for key, value in own.items():
        if value.device.type == "meta":
            raise ValueError("Load into a materialized CPU model before FSDP placement")
        if value.numel():
            shared[(value.device, value.data_ptr(), value.shape, value.stride())].add(key)
    present = set(shapes) - mismatched
    for names in shared.values():
        if names & present:
            present.update(names)
    # Explicit safetensors metadata may restore aliases in an untied architecture.
    for key, source in aliases.items():
        if source in present and shapes[source] == tuple(own[key].shape):
            present.add(key)
    missing = own.keys() - present
    rejected = {key for key in missing | mismatched if not key.startswith(tuple(reinitialize))}
    if rejected:
        raise ValueError(f"Missing or incompatible checkpoint weights: {sorted(rejected)}")
    with torch.no_grad():
        for shard, _ in files:
            with safe_open(shard, framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    name = local_name(key)
                    if name not in mismatched and not name.startswith(tuple(exclude)):
                        value = stream.get_tensor(key)
                        own[name].copy_(value)
                        for alias, source in aliases.items():
                            if source == name and alias not in mismatched:
                                own[alias].copy_(value)
    return sorted(missing | mismatched)
