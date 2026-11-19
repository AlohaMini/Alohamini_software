"""Default decoding can fall back; explicit checkpoint/user settings remain authoritative."""

import builtins

import pytest

from alohamini.learning import loading


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_installed_but_unloadable_decoder_uses_pyav(monkeypatch, caplog, error):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "torchcodec.decoders":
            raise error("native FFmpeg library missing")
        return original(name, *args, **kwargs)

    loading._torchcodec_usable.cache_clear()
    monkeypatch.setattr(builtins, "__import__", unavailable)
    monkeypatch.setattr(loading.importlib.util, "find_spec", lambda _: object())
    try:
        assert loading.resolve_data_pipeline({})["video_backend"] == "pyav"
        assert loading.resolve_data_pipeline({"video_backend": "torchcodec"}) == {
            "video_backend": "torchcodec", "camera_workers": 3, "video_cache_size": 32,
            "return_uint8": True, "dataloader_multiprocessing_context": "spawn",
        }
        assert "native FFmpeg library missing" in caplog.text
    finally:
        loading._torchcodec_usable.cache_clear()


def test_loading_decoder_is_checked_once_per_process(monkeypatch):
    loading._torchcodec_usable.cache_clear()
    original = builtins.__import__

    def fail_if_reimported(name, *args, **kwargs):
        if name == "torchcodec.decoders":
            pytest.fail("re-imported")
        return original(name, *args, **kwargs)

    try:
        assert loading._torchcodec_usable()
        monkeypatch.setattr(builtins, "__import__", fail_if_reimported)
        assert loading._torchcodec_usable()
    finally:
        loading._torchcodec_usable.cache_clear()
