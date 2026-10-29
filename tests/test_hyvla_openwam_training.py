"""Native dataset -> shared trainer -> portable inference for both new policies."""

import pytest
import torch
from test_hyvla_openwam import tiny_options, tokenizer_assets
from test_native_learning import recording as make_recording


@pytest.fixture
def recording(tmp_path):
    return make_recording.__wrapped__(tmp_path)


@pytest.mark.parametrize("kind", ["hyvla05", "openwam"])
def test_shared_training_and_checkpoint(kind, recording, tmp_path):
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.policy import NativePolicy, save_checkpoint
    from alohamini.learning.train import train
    from alohamini.learning.train_config import parse_training_args
    from alohamini.policies.registry import algorithm

    torch.set_num_threads(2)
    assets = tmp_path / "assets"
    tokenizer_assets(assets)
    component = algorithm(kind)
    options = tiny_options(kind, assets)
    if kind == "hyvla05":
        options["network"].update(max_state_dim=32, max_action_dim=32)
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        cameras=["forward"],
        image_size=(32, 32),
        **component.sample_spec(options, cameras=["forward"]),
    )
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    model = component.build(options)
    stats = component.statistics(samples)
    base = tmp_path / "base"
    save_checkpoint(base, model, stats, samples, training={"mixed_precision": "bfloat16"})
    run = tmp_path / "run"
    cfg = dict(
        policy=kind,
        dataset=str(recording),
        pretrained_path=str(base),
        output_dir=str(run),
        device="cpu",
        train_episodes=[0],
        val_episodes=[1],
        steps=2,
        batch_size=1,
        image_size=[32, 32],
        num_workers=0,
        save_freq=1,
        log_freq=0,
        eval_steps=1,
        mixed_precision="bfloat16",
        ema={"enable": True, "decay": 0.9},
        scheduler={"type": "none"},
    )
    train(cfg)
    policy = NativePolicy(run / "checkpoint", device="cpu", offline=True)
    item = samples[0]
    # Dataset windows remain within the episode, and carry supervision masks.
    images = item["observation.images.forward"]
    if kind == "hyvla05":
        torch.testing.assert_close(images[0], images[-1], rtol=0, atol=0)
        assert item["observation.images.forward_is_pad"].tolist() == [True, False]
    else:
        assert item["observation.images.forward_is_pad"].tolist() == [False] + [True] * 4
    assert item["action_is_pad"].tolist() == [False] * min(model.config.chunk_size, 4) + [
        True
    ] * max(model.config.chunk_size - 4, 0)
    output = policy.predict(item)
    assert output.shape == (model.config.chunk_size, samples.output_features["action"].shape[0])
    assert output.isfinite().all()
    assert policy.manifest["training"]["paper_preset"]["overrides"]
    assert policy.manifest["config"]["assets_path"] == "assets"
    resume_config, _ = parse_training_args(
        [
            "--config_path",
            str((run / "checkpoint" / "train_config.json").resolve()),
            "--resume=true",
            "--steps=3",
        ]
    )
    train(resume_config)
    resumed = NativePolicy(run / "checkpoint", device="cpu", offline=True)
    assert resumed.predict(item).isfinite().all()
