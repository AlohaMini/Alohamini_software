"""Pi0 keeps continuous state conditioning and its own normalization protocol."""

import numpy as np
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.pi0.processor_pi0 import PI0Processor
from alohamini.policies.registry import algorithm


def tiny_options(tmp_path, monkeypatch):
    from test_lerobot_algorithms import small_pi05

    from alohamini.policies.pi05 import tokenizer

    options = small_pi05(tmp_path, monkeypatch)

    class Tokenizer:
        def __init__(self, *args):
            pass

        def tokenize(self, prompt, state=None):
            assert state is None  # Continuous state must never be discretized into Pi0 text.
            return np.array([2, 3]), np.array([True, True])

    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", Tokenizer)
    return options


def test_continuous_state_expert_backward_and_native_checkpoint(recording, tmp_path, monkeypatch):
    recipe = algorithm("pi0")
    options = tiny_options(tmp_path, monkeypatch)
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        image_size=(32, 32),
        state="joint_position,base_velocity,lift_height",
        **recipe.sample_spec(options, cameras=["forward"]),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data, options)
    model = recipe.build(options)
    batch = recipe.processor(model, stats, "cpu")(
        next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    )
    loss, outputs = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(outputs) == {"flow_loss"}
    assert model.network.state_proj.weight.grad is not None
    assert not model.network.pi05 and not hasattr(model.network, "time_mlp_in")
    checkpoint = tmp_path / "checkpoint"
    model.eval()
    save_checkpoint(checkpoint, model, stats, data, training={"tf32": False})
    loaded = NativePolicy(checkpoint, offline=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    prediction = loaded.model.predict_action_chunk(batch)
    assert prediction.shape == (2, 2, 32) and torch.isfinite(prediction).all()


def test_pi0_mean_std_inverse_and_relative_anchor_do_not_use_quantiles():
    stats = {key: {"mean": [3.0] * 18, "std": [2.0] * 18} for key in ("state", "actions")}
    processor = PI0Processor(stats, tokenizer=None)
    state = torch.arange(18.0)[None]
    values = torch.full((1, 2, 18), 7.0)
    encoded = processor.encode_actions(values, state)
    padded = torch.nn.functional.pad(encoded, (0, 14))
    torch.testing.assert_close(processor.restore_actions(padded, state), values)
