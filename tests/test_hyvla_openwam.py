"""Real reduced networks and shared data/checkpoint contracts; no Hub access."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from alohamini.policies.configuration import PolicyFeature
from alohamini.policies.registry import algorithm


def tokenizer_assets(path):
    vocab = {"[PAD]": 0, "[UNK]": 1, "move": 2, "cube": 3, "[EOS]": 4}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(path)


def tiny_options(kind, path):
    features = {
        "observation.state": PolicyFeature("STATE", (3,)),
        "observation.images.head": PolicyFeature("VISUAL", (3, 32, 32)),
    }
    common = dict(
        assets_path=str(path),
        input_features=features,
        output_features={"action": PolicyFeature("ACTION", (3,))},
    )
    if kind == "hyvla05":
        return dict(
            **common,
            chunk_size=2,
            n_action_steps=1,
            history_frames=2,
            history_stride=1,
            network=dict(
                vlm_config_dict=dict(
                    model_type="hunyuan_vl_mot",
                    text_config=dict(
                        hidden_size=32,
                        intermediate_size=64,
                        vocab_size=8,
                        org_vocab_size=8,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=8,
                        attention_head_dim=8,
                        pad_token_id=0,
                        bos_token_id=1,
                        eos_token_id=4,
                        max_position_embeddings=1024,
                    ),
                    vision_config=dict(
                        native_vit=dict(
                            width=32, layers=1, heads=4, mlp_ratio=2, global_pool="", patch_size=16
                        ),
                        native_position_size=32,
                    ),
                ),
                proj_width=32,
                max_state_dim=4,
                max_action_dim=4,
                resize_imgs_with_padding=(32, 32),
                tokenizer_max_length=8,
                num_steps=2,
                use_video_encoder=True,
                spacetime_layer_stride=1,
            ),
        )
    from alohamini.policies.openwam.configuration_openwam import WAN_DEFAULTS

    return dict(
        **common,
        chunk_size=16,
        n_action_steps=1,
        canvas_size=(96, 32),
        num_steps=2,
        action_hidden_dim=32,
        action_ffn_dim=64,
        wan={
            **WAN_DEFAULTS,
            "dim": 32,
            "ffn_dim": 64,
            "num_heads": 4,
            "num_layers": 1,
            "freq_dim": 16,
            "text_dim": 32,
        },
        text_encoder=dict(
            vocab=8, dim=32, dim_attn=32, dim_ffn=64, num_heads=4, num_layers=1, dropout=0
        ),
        vae=dict(dim=8),
    )


def raw_batch(kind, config):
    count = 2 if kind == "hyvla05" else 5
    return {
        "observation.state": torch.randn(1, 3),
        "observation.images.head": torch.rand(1, count, 3, 32, 32),
        "observation.images.head_is_pad": torch.zeros(1, count, dtype=torch.bool),
        "action": torch.randn(1, config.chunk_size, 3),
        "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool),
        "task": ["move cube"],
    }


class NativeNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_network(self, kind):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            tokenizer_assets(path)
            component = algorithm(kind)
            model = component.build(tiny_options(kind, path))
            self.check_base_import(kind, path, component, model)
            batch = raw_batch(kind, model.config)
            batch = {
                key: value.repeat(2, *([1] * (value.ndim - 1)))
                if isinstance(value, torch.Tensor)
                else value * 2
                for key, value in batch.items()
            }
            model.train()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss, metrics = model(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters()
                    if p.requires_grad
                )
            )
            optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
            optimizer.step()
            model.eval()
            inference = {k: v for k, v in batch.items() if not k.startswith("action")}
            if kind == "openwam":
                inference["observation.images.head"] = batch["observation.images.head"][:, 0]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                torch.manual_seed(12)
                expected = model.predict_action_chunk(inference)
            self.assertEqual(expected.shape, (2, model.config.chunk_size, 3))
            self.assertTrue(torch.isfinite(expected).all())
            if kind == "openwam":
                changed = dict(batch)
                changed["observation.images.head"] = batch["observation.images.head"].clone()
                changed["observation.images.head"][:, 1:] = 0
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    torch.manual_seed(12)
                    current_only = model.predict_action_chunk(changed)
                torch.testing.assert_close(current_only, expected, rtol=0, atol=0)
            from dataclasses import asdict

            manifest = {"config": asdict(model.config)}
            target = path / "saved"
            target.mkdir()
            component.save(target, model, manifest)
            restored = component.build(
                component.checkpoint_options(target, manifest["config"], "cpu", None, "checkpoint")
            )
            component.load(target, restored)
            restored.eval()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                torch.manual_seed(12)
                actual = restored.predict_action_chunk(inference)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def check_base_import(self, kind, path, component, model):
        from safetensors.torch import save_file

        original = {key: value.clone() for key, value in model.network.state_dict().items()}
        source = {key: value.clone() + 0.01 for key, value in original.items()}
        if kind == "hyvla05":
            # Released HF weights retain the LM head alias, not the tied embedding alias.
            embedding = "model.dual_tower.vlm.model.language_model.model.embed_tokens.weight"
            source.pop(embedding)
            self.assertNotIn("model.dual_tower.expert.lm_head.weight", source)
            filename, heads = "model.safetensors", ()
        else:
            filename = "checkpoint_step_000001.safetensors"
            heads = (
                "action_backbone.action_encoder.",
                "action_backbone.action_decoder.",
                "proprio_encoder.",
            )
        save_file(source, path / filename)
        component.initialize(model, {"pretrained_path": str(path)})
        for key, actual in model.network.state_dict().items():
            if key.startswith(heads):
                torch.testing.assert_close(actual, original[key], rtol=0, atol=0)
            elif key in source:
                torch.testing.assert_close(actual, source[key], rtol=0, atol=0)

    def test_hyvla_real_network(self):
        self.check_network("hyvla05")

    def test_openwam_real_network(self):
        self.check_network("openwam")

    def test_local_assets_follow_workspace(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("os.environ", ALOHAMINI_WORKSPACE=tmp),
        ):
            for kind in ("hyvla05", "openwam"):
                path = Path(tmp) / "pretrained" / kind
                if kind == "openwam":
                    path = path / "tokenizer/google/umt5-xxl"
                tokenizer_assets(path)
                supplied = tiny_options(kind, path)
                supplied.pop("assets_path")
                options = algorithm(kind).options({"model": supplied}, "cpu")
                self.assertEqual(Path(options["assets_path"]), path)

    def test_invalid_temporal_and_representation_contracts(self):
        for kind in ("hyvla05", "openwam"):
            component = algorithm(kind)
            options = tiny_options(kind, Path("unused"))
            with self.assertRaises(ValueError):
                component.config_class(**{**options, "n_action_steps": 100})
            with self.assertRaises(ValueError):
                component.config_class(**{**options, "action_representation": "auto"})
        for key in ("shift", "lambda_video", "lambda_action"):
            with self.assertRaises(ValueError):
                component.config_class(**{**options, key: float("nan")})
        with self.assertRaises(ValueError):
            component.config_class(**{**options, "chunk_size": 17})
        with self.assertRaises(ValueError):
            component.config_class(**{**options, "camera_layout": ["unknown", None, None]})


if __name__ == "__main__":
    unittest.main()
