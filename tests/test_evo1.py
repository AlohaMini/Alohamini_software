"""Native EVO1 stage boundaries and reduced real InternVL/action Transformers."""

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.evo1.adapter import Evo1Algorithm


def tiny_options(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        InternVLConfig,
        InternVLVisionConfig,
        PreTrainedTokenizerFast,
        Qwen2Config,
    )

    tmp_path.mkdir(exist_ok=True, parents=True)
    vocab = {
        token: i
        for i, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "<IMG_CONTEXT>",
                "<img>",
                "</img>",
                "test",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        additional_special_tokens=["<IMG_CONTEXT>", "<img>", "</img>"],
    )
    tokenizer.save_pretrained(tmp_path)
    config = InternVLConfig(
        vision_config=InternVLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=32,
            patch_size=4,
        ),
        text_config=Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            pad_token_id=0,
        ),
        image_token_id=3,
        image_seq_length=16,
        downsample_ratio=0.5,
    )
    config.save_pretrained(tmp_path)
    return dict(
        vlm_model_name=str(tmp_path),
        vlm_num_layers=1,
        image_resolution=[32, 32],
        dtype="float32",
        embed_dim=32,
        hidden_dim=64,
        state_hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_inference_timesteps=2,
        chunk_size=2,
        n_action_steps=1,
        max_text_length=128,
        max_views=1,
    )


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_evo1_stage_gradients_and_native_checkpoint(recording, tmp_path, stage):
    recipe = Evo1Algorithm()
    options = {**tiny_options(tmp_path / "assets"), "training_stage": stage}
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options)
    stats = recipe.statistics(data)
    processor = recipe.processor(model, stats, "cpu")
    raw = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    batch = processor(raw)
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    vlm = model.model.embedder.model
    assert any(p.grad is not None for p in vlm.parameters()) == (stage == "stage2")
    assert len(vlm.language_model.layers) == 1
    model.eval()
    actions = processor.action(model.predict_action_chunk(batch))
    assert actions.shape == (2, 2, 18) and torch.isfinite(actions).all()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={})
    loaded = NativePolicy(checkpoint, offline=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
