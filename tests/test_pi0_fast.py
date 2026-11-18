"""Real reduced PaliGemma CE, official FAST codec, and native checkpoint assets."""

import json
from pathlib import Path

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.pi0_fast.adapter import PI0FastAlgorithm


def tiny_options(tmp_path, monkeypatch):
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
    from tokenizers import ByteLevelBPETokenizer

    from alohamini.policies.pi05 import gemma_pytorch

    tmp_path.mkdir(parents=True, exist_ok=True)
    proto = sp_pb2.ModelProto()
    proto.trainer_spec.vocab_size = 2048
    proto.trainer_spec.pad_id = 3
    proto.normalizer_spec.name = "identity"
    proto.normalizer_spec.add_dummy_prefix = False
    proto.normalizer_spec.remove_extra_whitespaces = False
    for i, piece in enumerate(
        ["<unk>", "<s>", "</s>", "<pad>", "▁"] + [chr(i) for i in range(33, 127)]
    ):
        item = proto.pieces.add(piece=piece, score=-1)
        item.type = 2 if i == 0 else 3 if i < 4 else 1
    while len(proto.pieces) < 2048:
        proto.pieces.add(piece=f"unused{len(proto.pieces)}", score=-100, type=1)
    sp_path = tmp_path / "tokenizer.model"
    sp_path.write_bytes(proto.SerializeToString())
    fast = tmp_path / "fast"
    fast.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["".join(map(chr, range(256)))], vocab_size=256)
    bpe._tokenizer.save(str(fast / "tokenizer.json"))
    (fast / "processor_config.json").write_text(
        json.dumps(dict(scale=10, min_token=-100, vocab_size=256))
    )
    original = gemma_pytorch.PaliGemmaForConditionalGeneration

    def paligemma(config):
        config.text_config.vocab_size = 2048
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 112
        return original(config)

    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", paligemma)
    return dict(
        dtype="float32",
        paligemma_variant="dummy",
        chunk_size=2,
        n_action_steps=1,
        tokenizer_path=str(sp_path),
        fast_tokenizer_path=str(fast),
        max_token_len=200,
        fast_max_tokens=256,
        max_decoding_steps=4,
        ce_chunk_size=32,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fast_ce_codec_checkpoint_and_malformed_prediction(
    recording, tmp_path, monkeypatch, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe = PI0FastAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data, options)
    model = recipe.build(options).to(device)
    processor = recipe.processor(model, stats, device)
    batch = processor(next(iter(torch.utils.data.DataLoader(data, batch_size=2))))
    assert batch["observation"].state.shape[-1] == batch["action"].shape[-1] == 18
    assert model.network.gemma_expert is None
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(metrics) == {"fast_loss"}
    assert model.network.paligemma.language_model.layers[0].self_attn.q_proj.weight.grad is not None
    sequences = [
        row[mask].tolist()
        for row, mask in zip(batch["fast_tokens"], batch["fast_mask"], strict=True)
    ]
    decoded = processor.decode(sequences, device=device)
    # DCT rounding gives <= .05 L2 coefficient error per coordinate.
    assert (decoded - batch["action"]).square().sum(1).sqrt().max() <= 0.05 * 2**0.5 + 1e-5
    with pytest.raises(ValueError, match="prefix"):
        processor.decode([[2]], device="cpu")
    # Register only after this network/processor has passed its native smoke path.
    from alohamini.policies import registry

    monkeypatch.setitem(
        registry.ALGORITHMS, "pi0_fast", ("alohamini.policies.pi0_fast.adapter", "PI0FastAlgorithm")
    )
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={"tf32": False})
    loaded = NativePolicy(checkpoint, device=device, offline=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)
    assert Path(loaded.config.tokenizer_path).is_relative_to(checkpoint)
    # An untrained random decoder must reject incomplete actions rather than send zeros.
    with pytest.raises(ValueError, match="max_decoding_steps|prefix|terminator|coefficients"):
        loaded.predict(data[0])


def test_cached_generation_matches_full_teacher_forcing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from alohamini.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    from alohamini.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    options = tiny_options(tmp_path / "assets", monkeypatch)
    config = PI0FastConfig(
        **options,
        input_features={
            "observation.state": {"type": "STATE", "shape": [18]},
            "observation.images.forward": {"type": "VISUAL", "shape": [3, 224, 224]},
        },
        output_features={"action": {"type": "ACTION", "shape": [18]}},
    )
    model = PI0FastPolicy(config).eval()
    batch = {
        "observation": SimpleNamespace(
            images={"forward": torch.rand(2, 3, 224, 224) * 2 - 1},
            image_masks={"forward": torch.ones(2, dtype=torch.bool)},
            tokenized_prompt=torch.tensor([[1, 4, 5, 0], [1, 6, 0, 0]]),
            tokenized_prompt_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        )
    }
    with torch.no_grad():
        embeddings, valid = model._prefix(batch)
        hidden, cache = model._transform(
            embeddings,
            valid[:, :, None] & valid[:, None, :],
            valid.long().cumsum(-1) - 1,
            use_cache=True,
        )
        token = torch.tensor([[10], [11]])
        extended = torch.cat((valid, torch.ones(2, 1, dtype=torch.bool)), dim=1)
        step, _ = model._transform(
            model._embed_tokens(token),
            extended[:, None, :],
            valid.sum(-1, keepdim=True),
            cache=cache,
            use_cache=True,
        )
        mask = extended[:, :, None] & extended[:, None, :]
        mask[:, :-1, -1] = False
        full, _ = model._transform(
            torch.cat((embeddings, model._embed_tokens(token)), dim=1),
            mask,
            extended.long().cumsum(-1) - 1,
        )
    torch.testing.assert_close(step[:, -1], full[:, -1], rtol=1e-4, atol=1e-5)
