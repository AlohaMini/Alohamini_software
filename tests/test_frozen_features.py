"""Generic prepared features must be atomic, immutable and bounded in memory."""

import json

import pytest
import torch

from alohamini.learning.frozen_features import (
    FrozenFeatureCache,
    prepare_features,
    source_signature,
)


def test_prepared_features_reuse_copies_and_detect_modified_contents(tmp_path):
    calls = []

    def encode(keys):
        calls.append(tuple(keys))
        return [torch.full((2, 3), float(ord(key))) for key in keys]

    path = prepare_features(
        tmp_path, {"encoder": ("revision", 1)}, ["b", "a", "b"], encode, batch_size=1
    )
    same = prepare_features(
        tmp_path,
        {"encoder": ("revision", 1)},
        ["a", "b"],
        lambda keys: pytest.fail("warm store must not encode"),
    )
    assert same == path and len(calls) == 2
    cache = FrozenFeatureCache(path, max_bytes=24)
    values = cache.get_many(
        ["b", "a", "b"], lambda keys: pytest.fail("prepared keys must not encode")
    )
    assert cache._bytes <= 24
    values[0].fill_(0)
    assert cache.get_many(["b"], encode)[0].mean() == 98
    manifest = json.loads((path / "features.json").read_text())
    tensor = path / manifest["entries"]["a"]["file"]
    raw = bytearray(tensor.read_bytes())
    raw[-1] ^= 1
    tensor.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        FrozenFeatureCache(path).get_many(["a"], encode)


def test_failed_producer_never_publishes_partial_store(tmp_path):
    def fail(keys):
        if keys == ["b"]:
            raise RuntimeError("encoder failed")
        return [torch.ones(2)]

    with pytest.raises(RuntimeError, match="encoder failed"):
        prepare_features(tmp_path, {"source": "test"}, ["a", "b"], fail, batch_size=1)
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]
    path = prepare_features(
        tmp_path, {"source": "test"}, ["a", "b"], lambda keys: [torch.ones(2) for k in keys]
    )
    assert len(FrozenFeatureCache(path).manifest["entries"]) == 2
    with pytest.raises(ValueError, match="Trainable"):
        prepare_features(tmp_path, {}, ["x"], lambda keys: [torch.ones(2, requires_grad=True)])


def test_dynamic_cache_calls_only_for_missing_keys_and_is_bounded():
    calls = []

    def encode(keys):
        calls.append(tuple(keys))
        return [torch.full((4,), float(len(k))) for k in keys]

    cache = FrozenFeatureCache(max_bytes=16)
    first = cache.get_many(["one", "one"], encode)
    second = cache.get_many(["one"], encode)
    assert calls == [("one",)]
    first[0].fill_(999)
    assert second[0].mean() == 3
    cache.get_many(["four"], encode)
    assert cache._bytes == 16
    cache.get_many(["one"], encode)
    assert len(calls) == 3


def test_source_signature_detects_local_encoder_changes(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    initial = source_signature([tmp_path])
    path.write_text('{"changed":true}')
    assert source_signature([tmp_path]) != initial
