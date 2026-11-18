"""Real small Gemma3 VLM/expert, native masks, preprocessing and checkpoint assets."""

from contextlib import nullcontext

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.dm05.adapter import DM05Algorithm


def tiny_options(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        Gemma3Config,
        Gemma3ImageProcessor,
        Gemma3Processor,
        Gemma3TextConfig,
        PreTrainedTokenizerFast,
        SiglipVisionConfig,
    )

    path.mkdir(parents=True, exist_ok=True)
    names = ["[PAD]", "[UNK]", "[BOS]", "<image>", "<boi>", "<eoi>", "[EOS]", "test"]
    backend = Tokenizer(
        models.WordLevel({name: i for i, name in enumerate(names)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        extra_special_tokens=dict(image_token="<image>", boi_token="<boi>", eoi_token="<eoi>"),
    )
    template = (
        "{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type']=='image' %}<boi> {% else %}{{ item['text'] }} {% endif %}"
        "{% endfor %}{% endfor %}{% if add_generation_prompt %}action {% endif %}"
    )
    Gemma3Processor(
        image_processor=Gemma3ImageProcessor(size=dict(height=32, width=32), do_pan_and_scan=False),
        tokenizer=tokenizer,
        image_seq_length=4,
        chat_template=template,
    ).save_pretrained(path)
    common = dict(
        vocab_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        query_pre_attn_scalar=8,
        pad_token_id=0,
        max_position_embeddings=512,
        sliding_window=64,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters=dict(
            sliding_attention=dict(rope_type="default", rope_theta=10000.0),
            full_attention=dict(rope_type="default", rope_theta=10000.0),
        ),
    )
    text = Gemma3TextConfig(hidden_size=32, **common)
    action = Gemma3TextConfig(hidden_size=16, **common)
    vlm = Gemma3Config(
        text_config=text,
        vision_config=SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=32,
            patch_size=8,
        ),
        mm_tokens_per_image=4,
        image_token_index=3,
        boi_token_index=4,
        eoi_token_index=5,
    )
    return dict(
        assets_path=str(path),
        dtype="float32",
        chunk_size=2,
        n_action_steps=1,
        diffusion_steps=2,
        core_config=dict(
            vlm_config=vlm.to_dict(), action_config=action.to_dict(), action_dim=32, chunk_size=2
        ),
    )


@pytest.mark.parametrize(
    "device,storage,precision",
    [("cpu", "preserve", "none"), ("cuda", "preserve", "none"), ("cuda", "float32", "bfloat16")],
)
def test_dm05_native_forward_padding_and_checkpoint(
    recording, tmp_path, monkeypatch, device, storage, precision
):
    from alohamini.learning.precision import apply_trainable_dtype
    from alohamini.policies import registry

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    recipe = DM05Algorithm()
    options = tiny_options(tmp_path / "assets")
    options["dtype"] = "bfloat16" if device == "cuda" else "float32"
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    stats = recipe.statistics(samples, options)
    model = apply_trainable_dtype(recipe.build(options), storage).to(device)
    processor = recipe.processor(model, stats, device)

    def amp():
        return (
            torch.autocast(device, dtype=torch.bfloat16)
            if precision == "bfloat16"
            else nullcontext()
        )

    batch = processor(next(iter(torch.utils.data.DataLoader(samples, batch_size=2))))
    batch["action_is_pad"] = torch.tensor([[False, True], [False, False]], device=device)
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
    with amp():
        loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"flow_loss"}
    assert model.model.model.action_out_proj.weight.grad is not None
    assert model.model.model.language_model.layers[0].self_attn.q_proj.weight.grad is not None
    assert (
        len(model.model.model.language_model.layers)
        == len(model.model.model.action_expert.layers)
        == 2
    )
    assert model.model.model.vlm.model.language_model.embed_tokens.weight.grad is None
    modified = {**batch, "action": batch["action"].clone()}
    modified["action"][0, 1] = 1e6
    torch.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    with amp():
        expected, _ = model(modified)
    torch.testing.assert_close(loss, expected)
    model.eval()
    with amp():
        actions = processor.action(model.predict_action_chunk(batch))
    assert actions.shape == (2, 2, 18) and torch.isfinite(actions).all()
    monkeypatch.setitem(
        registry.ALGORITHMS, "dm05", ("alohamini.policies.dm05.adapter", "DM05Algorithm")
    )
    path = tmp_path / "checkpoint"
    save_checkpoint(
        path, model, stats, samples, training={"tf32": False, "mixed_precision": precision}
    )
    loaded = NativePolicy(path, device=device, offline=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    assert loaded.predict(samples[0]).shape == (2, 18)
    if device == "cpu":
        from safetensors.torch import save_file

        original = tmp_path / "raw-opendm"
        original.mkdir()
        legacy = {
            key.replace(
                "model.vlm.model.vision_tower.", "model.vlm.model.vision_tower.vision_model."
            ): value.detach().clone().contiguous()
            for key, value in model.model.state_dict().items()
        }
        save_file(legacy, original / "model.safetensors")
        target = recipe.build(options)
        recipe.initialize(target, dict(pretrained_path=str(original)))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
