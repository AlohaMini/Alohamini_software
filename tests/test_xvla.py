"""Real reduced Florence/action Transformers with native parameter groups and checkpoints."""

import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.xvla.adapter import XVLAAlgorithm


def tiny_options(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        BartConfig,
        CLIPImageProcessor,
        Florence2Config,
        Florence2VisionConfig,
        PreTrainedTokenizerFast,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "test": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(tmp_path)
    CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    ).save_pretrained(tmp_path)
    config = Florence2Config(
        vision_config=Florence2VisionConfig(
            depths=[1],
            patch_size=[3],
            patch_stride=[4],
            patch_padding=[1],
            patch_prenorm=[False],
            embed_dim=[32],
            num_heads=[4],
            num_groups=[4],
            window_size=2,
            projection_dim=32,
            max_position_embeddings=32,
        ),
        text_config=BartConfig(
            vocab_size=32,
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_position_embeddings=256,
            pad_token_id=0,
        ),
    )
    return dict(
        assets_path=str(tmp_path),
        florence_config=config.to_dict(),
        chunk_size=2,
        n_action_steps=1,
        hidden_size=32,
        depth=1,
        num_heads=4,
        num_domains=2,
        len_soft_prompts=2,
        dim_time=8,
        max_len_seq=256,
        tokenizer_max_length=16,
        num_denoising_steps=2,
    )


def test_xvla_native_backward_domain_groups_and_checkpoint(recording, tmp_path, monkeypatch):
    from alohamini.policies import registry

    recipe = XVLAAlgorithm()
    options = tiny_options(tmp_path / "assets")
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    model = recipe.build(options)
    stats = recipe.statistics(samples)
    processor = recipe.processor(model, stats, "cpu")
    raw = next(iter(torch.utils.data.DataLoader(samples, batch_size=2)))
    batch = processor(raw)
    loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"joints_loss"}
    assert model.model.transformer.soft_prompt_hub.weight.grad is not None
    assert len(model.model.transformer.blocks) == 1
    assert len(model.model.vlm.language_model.encoder.layers) == 1
    assert not hasattr(model.model.vlm.language_model, "decoder")
    assert {group["name"] for group in model.get_optim_params()} == {
        "vlm",
        "transformer_core",
        "soft_prompts",
        "action_heads",
    }
    model.eval()
    actions = processor.action(model.predict_action_chunk(batch))
    assert actions.shape == (2, 2, 18) and torch.isfinite(actions).all()
    monkeypatch.setitem(
        registry.ALGORITHMS, "xvla", ("alohamini.policies.xvla.adapter", "XVLAAlgorithm")
    )
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, stats, samples, training={"tf32": False})
    loaded = NativePolicy(path, offline=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)
    assert loaded.predict(samples[0]).shape == (2, 18)
