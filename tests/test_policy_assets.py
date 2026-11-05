"""Local assets fail locally; explicit Hub IDs respect the selected offline mode."""

import json
import os
from pathlib import Path

import httpx
import huggingface_hub.constants
import pytest
import requests
from test_acdit import siglip_assets as siglip_assets
from test_diffusion_fastwam import features
from test_gr00t_lewm import groot_assets as groot_assets
from test_multi_task_dit import dit_options
from test_smolvla import tiny_assets as tiny_assets


@pytest.fixture
def forbid_hub_http(monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    for name in os.environ:
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Runtime attempted an HTTP request for missing assets")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(requests.Session, "send", forbidden)
    return attempts


@pytest.mark.parametrize("encoder", ["vision", "text"])
@pytest.mark.parametrize("load_weights", [True, False])
@pytest.mark.parametrize("source", ["uncached_repo", "missing_directory", "empty_directory"])
def test_clip_missing_assets_fail_without_http(
    tmp_path, forbid_hub_http, monkeypatch, encoder, load_weights, source
):
    from alohamini.policies.multi_task_dit.modeling_multi_task_dit import (
        CLIPTextEncoder,
        CLIPVisionEncoder,
    )

    name = str(tmp_path / "absent")
    if source == "uncached_repo":
        name = f"alohamini-missing-assets/{tmp_path.name}"
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    elif source == "empty_directory":
        name = str(tmp_path)
    cls = CLIPVisionEncoder if encoder == "vision" else CLIPTextEncoder
    with pytest.raises(OSError):
        cls(name, load_weights=load_weights)
    assert not forbid_hub_http


def test_explicit_clip_repo_can_request_online_resources(tmp_path, forbid_hub_http):
    from alohamini.policies.multi_task_dit.modeling_multi_task_dit import CLIPVisionEncoder

    # Intercept the HTTP boundary; prove an explicit repo ID is not forced offline.
    with pytest.raises((AssertionError, OSError)):
        CLIPVisionEncoder(f"alohamini-missing-assets/{tmp_path.name}")
    assert forbid_hub_http


def test_clip_missing_tokenizer_does_not_fetch_a_replacement(tmp_path, forbid_hub_http):
    from alohamini.policies.registry import algorithm

    options = dit_options(tmp_path / "assets")
    text = tmp_path / "assets" / "clip_text"
    # Keep a valid encoder config/weights but remove all tokenizer resources.
    for path in text.iterdir():
        if path.name != "config.json" and path.suffix != ".safetensors":
            path.unlink()
    config_path = text / "config.json"
    config = json.loads(config_path.read_text())
    config["_name_or_path"] = "openai/clip-vit-base-patch16"
    config_path.write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        algorithm("multi_task_dit").build({**features(), **options})
    assert not forbid_hub_http


@pytest.mark.parametrize("kind", ["acdit", "gr00t_n17", "hyvla05", "openwam", "smolvla", "spirit"])
@pytest.mark.parametrize("missing", ["directory", "tokenizer"])
def test_native_policy_missing_assets_do_not_use_hub(
    tmp_path, request, forbid_hub_http, kind, missing
):
    from alohamini.policies.registry import algorithm

    if kind == "acdit":
        from test_acdit import tiny_options

        path = request.getfixturevalue("siglip_assets")
        options = tiny_options(path, stage="full", points=False)
    elif kind == "gr00t_n17":
        from test_gr00t_lewm import options as groot_options

        path = request.getfixturevalue("groot_assets")
        options = groot_options(kind, path)
    elif kind == "smolvla":
        from test_smolvla import options as smol_options

        path = Path(request.getfixturevalue("tiny_assets"))
        options = smol_options(str(path))
    elif kind == "spirit":
        from test_spirit import tiny_spirit_options

        path = tmp_path / "assets"
        options = {**features(), **tiny_spirit_options(path)}
    else:
        from test_hyvla_openwam import tiny_options, tokenizer_assets

        path = tmp_path / "assets"
        tokenizer_assets(path)
        options = tiny_options(kind, path)
    if missing == "directory":
        path.rename(path.with_name(path.name + "-removed"))
    else:
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            (path / name).unlink(missing_ok=True)
    with pytest.raises((OSError, ValueError, TypeError)):
        algorithm(kind).build(options)
    assert not forbid_hub_http
