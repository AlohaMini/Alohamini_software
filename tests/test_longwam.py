"""Real reduced Wan/UMT5 networks through native data and checkpoint interfaces."""

import json

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.registry import algorithm


def tiny_options(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tmp_path.mkdir(exist_ok=True, parents=True)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "test": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(tmp_path)
    return dict(
        past_obs_steps=16,
        future_video_steps=16,
        action_horizon=2,
        n_action_steps=1,
        camera_keys=["observation.images.forward"],
        camera_size=(16, 16),
        action_dim=18,
        proprio_dim=18,
        dtype="float32",
        tokenizer_path=str(tmp_path),
        tokenizer_max_length=8,
        action_hidden_dim=16,
        action_ffn_dim=32,
        num_inference_steps=1,
        imagine_steps=1,
        video_config=dict(
            patch_size=(1, 1, 1),
            num_attention_heads=1,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
        ),
        vae_config=dict(
            base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        ),
        text_encoder_config=dict(
            vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2, dropout_rate=0.0
        ),
    )


def assembled(recording, tmp_path, **overrides):
    recipe = algorithm("longwam")
    options = {**tiny_options(tmp_path / "assets"), **overrides}
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=["forward"]),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    recipe.prepare_options(options, data)
    stats = recipe.statistics(data, options)
    model = recipe.build(options)
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(next(iter(torch.utils.data.DataLoader(data, batch_size=2))))
    return recipe, data, stats, model, processor, batch


@pytest.mark.parametrize("mode", ["idm", "cod"])
def test_native_longwam_backward_checkpoint_and_no_future_inference(recording, tmp_path, mode):
    recipe, data, stats, model, processor, batch = assembled(
        recording, tmp_path, denoising=mode, imagine_sigma=0.9 if mode == "idm" else 0.0
    )
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.dit.action.head.weight.grad is not None
    assert all(p.grad is None for p in model.vae.parameters())
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert set(metrics) == {"loss_video", "loss_action"}
    model.eval()
    expected = model.predict_action_chunk(batch)
    changed = {**batch, "observation.images.forward": batch["observation.images.forward"].clone()}
    current = model.config.past_obs_steps // 4
    changed["observation.images.forward"][:, current + 1 :] = -1
    torch.testing.assert_close(model.predict_action_chunk(changed), expected, rtol=0, atol=0)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={"tf32": False})
    restored = NativePolicy(checkpoint, offline=True)
    prepared = restored.processor(next(iter(torch.utils.data.DataLoader(data, batch_size=2))))
    torch.testing.assert_close(
        restored.model.predict_action_chunk(prepared), expected, rtol=0, atol=0
    )
    assert json.loads((checkpoint / "config.json").read_text())["dtype"] == "float32"


def test_longwam_global_sample_weight_and_layer_count(recording, tmp_path):
    recipe, data, stats, model, processor, batch = assembled(recording, tmp_path)
    state = torch.get_rng_state()
    full = model(batch)[0]
    torch.set_rng_state(state)
    weighted = model({**batch, "_loss_weight": 0.25})[0]
    torch.testing.assert_close(weighted, full * 0.25, rtol=0, atol=0)
    assert len(model.dit.action.blocks) == len(model.dit.video.blocks) == 1
    spec = recipe.inference_spec({}, cameras=["forward"])
    assert all(i <= 0 for k, v in spec["delta_indices"].items() if k != "action" for i in v)
