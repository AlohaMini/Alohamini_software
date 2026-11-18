"""Reduced Qwen2.5-VL EO-1 through native datasets and portable checkpoints."""

import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.eo1.adapter import EO1Algorithm


def tiny_options(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2_5_VLConfig,
        Qwen2_5_VLProcessor,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
    )

    from alohamini.policies.eo1.processor import SPECIAL_TOKENS

    tmp_path.mkdir(exist_ok=True, parents=True)
    special = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    vocab = {v: i for i, v in enumerate(special + SPECIAL_TOKENS + ["test", "longer", "task"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        additional_special_tokens=special[3:] + SPECIAL_TOKENS,
    )
    template = (
        "{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|> "
        "{% else %}{{ item['text'] }} {% endif %}{% endfor %}{% endfor %}"
    )
    processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=4, temporal_patch_size=2, merge_size=2, min_pixels=64, max_pixels=1024
        ),
        tokenizer=tokenizer,
        chat_template=template,
        video_processor=Qwen2VLVideoProcessor(),
    )
    processor.save_pretrained(tmp_path)
    config = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
            max_position_embeddings=512,
            rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[1, 1, 2]),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            fullatt_block_indexes=[0],
            window_size=16,
        ),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    return dict(
        assets_path=str(tmp_path),
        vlm_config=config.to_dict(),
        dtype="float32",
        chunk_size=2,
        n_action_steps=1,
        num_denoise_steps=2,
        image_min_pixels=64,
        image_max_pixels=1024,
        tokenizer_max_length=128,
    )


def test_eo1_native_backward_and_mixed_prompt_lengths(recording, tmp_path):
    recipe = EO1Algorithm()
    options = tiny_options(tmp_path / "assets")
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data)
    model = recipe.build(options)
    processor = recipe.processor(model, stats, "cpu")
    raw = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    raw["task"] = ["test", "test longer task"]
    batch = processor(raw)
    loss, outputs = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(outputs) == {"flow_loss"}
    assert model.model.action_in_proj.weight.grad is not None
    model.eval()
    actions = model.predict_action_chunk(batch)
    assert actions.shape == (2, 2, 18) and torch.isfinite(actions).all()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={})
    loaded = NativePolicy(checkpoint, offline=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
