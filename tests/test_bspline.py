"""Real BSP network, episode labels, independent statistics and checkpoint roundtrip."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from test_diffusion_fastwam import diffusion_options
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.data_fields import prepare_fields
from alohamini.learning.policy import NativePolicy, make_policy, save_checkpoint
from alohamini.policies.registry import algorithm


def test_two_arm_bsp_minibatch_and_checkpoint(recording, tmp_path):
    torch.set_num_threads(1)
    component = algorithm("bspline")
    options = {
        **diffusion_options(),
        "degree": 3,
        "horizon": 8,
        "n_action_steps": 1,
        "fit_tolerance": 0.01,
        "parameter_cache_dir": str(tmp_path / "cache"),
    }
    source = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **component.sample_spec(options, cameras=["forward"]),
    )
    data = prepare_fields(source, {}, component, options=options)
    options.update(
        input_features=deepcopy(data.input_features), output_features=deepcopy(data.output_features)
    )
    stats = component.statistics(data, options)
    assert len(stats["action"]["mean"]) == 18
    assert len(stats["trajectory"]["mean"]) == 19
    model = make_policy("bspline", options, stats)
    processor = component.processor(model, stats, "cpu")
    raw = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    batch = processor(raw)
    loss, metrics = model({**batch, **component.loss_counts(batch)})
    assert torch.isfinite(loss) and "diffusion_loss" in metrics
    loss.backward()
    assert model.generator.diffusion.unet.final_conv[-1].weight.grad is not None
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={"paper_preset": {"test": True}})
    native = NativePolicy(checkpoint, device="cpu", offline=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(native.model.state_dict()[key], value)
    # Supply a valid encoded sample to check the whole plan decoding boundary.
    native.model.predict_parameters = lambda prepared: batch["trajectory"][:1]
    plan = native._plan_from_batch({})
    assert len(plan.names) == 18 and plan.end_s > 0
    actual = np.asarray(plan.action_at(0.0))
    assert np.isfinite(actual).all()
    assert native.config.action_feature.shape == (18,)
    assert native.model.generator.config.action_feature.shape == (19,)


def test_validation_never_refits_training_parameter_statistics(recording, tmp_path):
    component = algorithm("bspline")
    options = dict(
        degree=3,
        horizon=8,
        n_obs_steps=2,
        n_action_steps=1,
        parameter_cache_dir=str(tmp_path / "cache"),
    )
    datasets = []
    for episode in (0, 1):
        source = AlohaMiniDataset(
            recording,
            episodes=[episode],
            state="joint_position,base_velocity,lift_height",
            image_size=(32, 32),
            **component.sample_spec(options, cameras=["forward"]),
        )
        datasets.append(prepare_fields(source, {}, component, options=options))
    train, validation = datasets
    stats = component.statistics(train, options)
    saved = deepcopy(stats)
    options.update(
        input_features=train.input_features,
        output_features=train.output_features,
        pretrained_backbone_weights=None,
        down_dims=[16, 32],
        crop_shape=None,
    )
    model = component.build(options)
    processor = component.processor(model, stats, "cpu")
    processor(next(iter(torch.utils.data.DataLoader(validation, batch_size=1))))
    assert stats == saved
    assert train.cache != validation.cache


def test_detached_bsp_trainer_validation_and_ema(recording, tmp_path, monkeypatch):
    from alohamini.learning.train import launch_training

    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    original = subprocess.Popen

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    options = {
        **diffusion_options(),
        "n_action_steps": 1,
        "parameter_cache_dir": str(tmp_path / "cache"),
    }
    job = launch_training(
        dict(
            dataset=str(recording),
            run_name="bsp-fixture",
            policy="bspline",
            device="cpu",
            num_workers=0,
            cpu_threads=1,
            train_episodes=[0],
            val_episodes=[1],
            state="joint_position,base_velocity,lift_height",
            image_size=[32, 32],
            steps=1,
            batch_size=2,
            eval_steps=1,
            model=options,
            scheduler={"type": "none"},
            review_note="Synthetic BSP integration fixture",
        )
    )
    try:
        code = processes[0].wait(timeout=60)
    finally:
        if processes[0].poll() is None:
            processes[0].terminate()
            processes[0].wait(timeout=5)
    assert code == 0, Path(job["log"]).read_text()
    run = Path(job["checkpoint"]).parent
    manifest = json.loads((run / "checkpoint/policy.json").read_text())
    assert manifest["training"]["train_episodes"] == [0]
    assert manifest["training"]["val_episodes"] == [1]
    assert (
        manifest["training"]["paper_preset"]["revision"]
        == algorithm("bspline").apply_preset({})["paper_preset"]["revision"]
    )
    assert len(manifest["stats"]["trajectory"]["min"]) == 19
    report = json.loads((run / "offline-evaluation.json").read_text())
    assert report["valid_action_steps"] > 0
    assert report["weights"] == "ema"
    assert NativePolicy(run / "checkpoint-ema", device="cpu").config.action_feature.shape == (18,)
