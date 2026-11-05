"""Official slots, calibrated FK/IK, and the native OpenWAM checkpoint boundary."""

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file
from scipy.spatial.transform import Rotation
from test_hyvla_openwam import tiny_options, tokenizer_assets
from test_kinematic_fields import descriptor, joint_state
from test_native_learning import recording as recording

from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.datasets.rotations import RotationEncoding
from alohamini.learning.checkpoint import write_checkpoint_metadata
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.data_fields import prepare_fields
from alohamini.learning.policy import NativePolicy, make_processor, save_checkpoint
from alohamini.learning.statistics import training_statistics
from alohamini.learning.train_config import parse_training_args
from alohamini.policies.openwam.semantic import (
    ROTATION,
    Alpha80Processor,
    action_padding,
    make_layout,
    scatter,
    semantic_statistics,
    validate_layout,
)
from alohamini.policies.registry import algorithm


def geometry():
    spec = descriptor()
    for key in ("source_names", "passthrough"):
        spec[key] = ["lift_axis.height_mm" if n == "lift.height" else n for n in spec[key]]
    return spec


def semantic_transform():
    return FieldTransforms(
        dict(
            state_representation="tcp",
            action_representation="tcp_absolute",
            rotation_representation=ROTATION,
            kinematic_fields=geometry(),
        )
    )


def layout():
    fields = semantic_transform()
    return make_layout(
        fields, {key: fields.geometry.feature() for key in ("observation.state", "action")}
    )


def test_columns_golden_and_backward_compatible_rows():
    matrices = Rotation.from_rotvec([[0.3, -0.7, 0.8], [np.pi, 0, 0], [0, 0, 0]]).as_matrix()
    columns = RotationEncoding(ROTATION)
    expected = np.concatenate([matrices[..., :, 0], matrices[..., :, 1]], axis=-1)
    np.testing.assert_array_equal(columns.encode(matrices), expected)
    np.testing.assert_allclose(columns.decode(expected), matrices, atol=1e-14)
    np.testing.assert_array_equal(
        RotationEncoding("rotation_6d").encode(matrices), matrices[..., :2, :].reshape(-1, 6)
    )
    with pytest.raises(ValueError, match="Degenerate"):
        columns.decode([1, 0, 0, 2, 0, 0])


def test_named_layout_and_pad_masks_do_not_guess_from_width():
    mapping = layout()
    slots = mapping["features"]["action"]["slots"]
    assert slots == [*range(10), *range(34, 44), *range(68, 72)]
    values = torch.arange(48.0).reshape(1, 2, 24)
    packed = scatter(values, slots)
    torch.testing.assert_close(packed[..., slots], values)
    assert packed[..., 10:34].count_nonzero() == 0
    mask = action_padding(torch.tensor([[False, True]]), values, slots)
    assert not mask[0, 0, slots].any() and mask[0, 1].all()
    assert mask[..., 10:34].all() and mask[..., 44:68].all() and mask[..., 72:].all()
    changed = deepcopy(mapping)
    changed["features"]["action"]["slots"][10] = 10
    with pytest.raises(ValueError, match="slot mapping"):
        validate_layout(changed)
    changed = deepcopy(mapping)
    changed["features"]["action"]["units"][-2] = "rad/s"
    with pytest.raises(ValueError, match="slot mapping"):
        validate_layout(changed)


def test_endpoint_normalization_constant_base_and_fk_ik_roundtrip():
    fields = semantic_transform()
    mapping = layout()
    current = joint_state()
    current[-4:-1] = 0
    targets = np.repeat(current[None], 2, axis=0)
    targets[:, 0] += [0.005, 0.01]
    targets[:, 7] -= [0.005, 0.01]
    targets[:, [6, 13]] += [[1, -1], [2, -2]]
    observed, actions = fields.encode(current, targets)
    raw = {
        key: {
            name: getattr(np, name)(array, axis=0).tolist()
            for name in ("min", "max", "mean", "std")
        }
        for key, array in {"observation.state": observed[None], "action": actions}.items()
    }
    stats = semantic_statistics(raw, mapping)
    config = SimpleNamespace(
        alpha80_layout=mapping,
        input_features={"observation.state": None},
        output_features={"action": None},
    )
    processor = Alpha80Processor(config, stats, "cpu")
    prepared = processor(
        {
            "observation.state": torch.tensor(observed[None]).float(),
            "action": torch.tensor(actions[None]).float(),
            "task": ["move cube"],
        }
    )
    # Grippers use physical closed/open endpoints, even with a narrow observed range.
    np.testing.assert_allclose(prepared["action"][0, :, 9], 2 * actions[:, 9] / 0.1 - 1, atol=1e-6)
    np.testing.assert_allclose(prepared["action"][0, :, 3:9], actions[:, 3:9], atol=1e-7)
    assert prepared["action"][..., 20:23].count_nonzero() == 0
    prediction = prepared["action"].clone()
    prediction[..., 20:24] = 12.5  # Includes static lift; residual must not become motion.
    physical = processor.action(prediction).double().numpy()[0]
    np.testing.assert_array_equal(physical[:, 20:23], 0)
    np.testing.assert_array_equal(physical[:, 23], current[-1])
    restored = fields.decode(current, physical)
    np.testing.assert_allclose(restored, targets, atol=2e-3)


def semantic_dataset(recording, tmp_path):
    root = tmp_path / "joint-radians"
    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    for path in (root / "meta/safety").glob("*.jsonl"):
        path.unlink()
    spec = geometry()
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        info["features"][key].update(names=spec["source_names"], shape=[18])
    info_path.write_text(json.dumps(info))
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            state = joint_state()
            state[0] += 0.002 * row["frame_index"]
            state[-4:-1] = 0
            target = state.copy()
            target[0] += 0.005
            target[7] -= 0.005
            target[6] += row["frame_index"]
            row["observation.state"], row["action"] = state.tolist(), target.tolist()
        for key in ("observation.state", "action"):
            table = table.set_column(
                table.schema.get_field_index(key),
                key,
                pa.array([row[key] for row in rows], type=pa.list_(pa.float32(), 18)),
            )
        pq.write_table(table, path)
    specification = tmp_path / "coordinates.json"
    specification.write_text(json.dumps(spec))
    component = algorithm("openwam")
    assets = tmp_path / "tokenizer"
    tokenizer_assets(assets)
    options = tiny_options("openwam", assets)
    options.update(action_representation="alpha80")
    settings = component.apply_preset(
        dict(model=options, data={"kinematics": {"spec": str(specification)}})
    )
    source = AlohaMiniDataset(
        root,
        episodes=[0],
        state="recorded",
        allow_unannotated=True,
        image_size=(32, 32),
        exclude_fixed_state=False,
        **component.sample_spec(options, cameras=["forward"]),
    )
    samples = prepare_fields(source, settings, component, options=options)
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    component.prepare_options(options, samples)
    return samples, settings, options, specification


def test_real_network_base_heads_train_checkpoint_and_host_decode(recording, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    samples, settings, options, spec = semantic_dataset(recording, tmp_path)
    component = algorithm("openwam")
    artifact = training_statistics(component, samples, options, "openwam")
    assert artifact["contract"]["data_fields"] == settings["data"]
    stats = artifact["stats"]
    model = component.build(options, stats)
    assert model.network.action_backbone.action_encoder.weight.shape[-1] == 80
    assert model.network.action_backbone.action_decoder.weight.shape[0] == 80
    assert model.network.proprio_encoder.weight.shape[-1] == 80
    # All base tensors, including the three semantic projections, must be loaded.
    state = {key: value.clone() + 0.01 for key, value in model.network.state_dict().items()}
    base = tmp_path / "base"
    base.mkdir()
    save_file(state, base / "checkpoint_step_000001.safetensors")
    component.initialize(model, {"pretrained_path": str(base)})
    for key, value in model.network.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    processor = make_processor(model, stats, "cpu", data_fields=settings["data"])
    prepared = processor(next(iter(torch.utils.data.DataLoader(samples, batch_size=2))))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = model(prepared)
    assert torch.isfinite(loss) and set(metrics) == {"loss_video", "loss_action"}
    loss.backward()
    grad = model.network.action_backbone.action_decoder.weight.grad
    assert grad[model.action_slots].abs().sum() > 0
    assert grad[model.inactive_slots].count_nonzero() == 0
    assert all(p.grad is None for p in model.network.video_backbone.vae.parameters())
    optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
    optimizer.step()
    model.eval()
    # Observe the actual denoising path: unused coordinates must follow sigma*noise.
    captured = []
    original = model.network.forward

    def track(action, t, **kwargs):
        captured.append((action.detach().clone(), t.detach().clone(), kwargs["proprio"].clone()))
        return original(action, t, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(model.network, "forward", track)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = model.predict_action_chunk(prepared)
    assert prediction.shape == (2, 16, 24) and torch.isfinite(prediction).all()
    first, second = captured
    # bfloat16 rounds both the timestep and intermediate update.
    expected = first[0][..., model.inactive_slots].float() * (5 / 6)
    torch.testing.assert_close(
        second[0][..., model.inactive_slots].float(), expected, atol=0.016, rtol=0.01
    )
    assert first[2][..., 10:34].count_nonzero() == 0
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    spec.unlink()
    shutil.rmtree(tmp_path / "tokenizer")
    loaded = NativePolicy(checkpoint, offline=True)
    assert loaded.config.alpha80_layout == options["alpha80_layout"]
    assert loaded.manifest["data_fields"] == settings["data"]
    assert any(op["operation"] == "openwam_alpha80" for op in loaded.manifest["field_transforms"])
    # Use a reachable predicted trajectory to test production unnormalization + seeded IK.
    # Random untrained network outputs are not claimed to be executable robot trajectories.
    loaded.model.predict_action_chunk = lambda batch: prepared["action"][:1]
    result = loaded.predict(samples[0])
    np.testing.assert_allclose(result.numpy(), samples[0]["_recorded_action"].numpy(), atol=2e-3)
    raw = samples.source[0]
    np.testing.assert_allclose(loaded.predict(raw).numpy(), result.numpy(), atol=1e-6)
    manifest_path = checkpoint / "policy.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["alpha80_layout"]["frames"]["left"] = "wrong_frame"
    write_checkpoint_metadata(checkpoint, manifest)
    with pytest.raises(ValueError, match="frozen FK/IK"):
        NativePolicy(checkpoint, offline=True)
    with pytest.raises(ValueError, match="Cannot switch"):
        component.options(
            {"pretrained_path": str(checkpoint), "model": {"action_representation": "recorded"}},
            "cpu",
        )


def test_preset_is_optional_and_rejects_incompatible_coordinates():
    component = algorithm("openwam")
    assert "data" not in component.apply_preset({})
    cfg = component.apply_preset({"model": {"action_representation": "alpha80"}})
    assert cfg["data"]["rotation_representation"] == ROTATION
    assert cfg["data"]["action_representation"] == "tcp_absolute"
    assert cfg["paper_preset"]["name"] == "openwam-alpha-semantic80-v1"
    assert component.apply_preset(cfg)["data"] == cfg["data"]
    with pytest.raises(ValueError, match="rotation_representation"):
        component.apply_preset(
            {
                "model": {"action_representation": "alpha80"},
                "data": {"rotation_representation": "rotation_6d"},
            }
        )
    with pytest.raises(ValueError, match="requires|require"):
        component.apply_preset({"data": {"action_representation": "tcp_absolute"}})


def test_cli_and_shared_training_checkpoint(recording, tmp_path):
    samples, settings, options, specification = semantic_dataset(recording, tmp_path)
    component = algorithm("openwam")
    base = tmp_path / "base"
    base.mkdir()
    model = component.build(options)
    save_file(model.network.state_dict(), base / "checkpoint_step_000001.safetensors")
    config_file = tmp_path / "training.json"
    config_file.write_text(
        json.dumps(
            dict(
                policy="openwam",
                dataset=str(samples.root),
                pretrained_path=str(base),
                model=asdict(model.config),
                data=settings["data"],
                device="cpu",
                cpu_threads=2,
                image_size=[32, 32],
                batch_size=2,
                num_workers=0,
                persistent_workers=False,
                train_episodes=[0],
                val_episodes=[],
                allow_unannotated=True,
                steps=2,
                save_freq=1,
                log_freq=0,
                output_dir=str(tmp_path / "run"),
            )
        )
    )
    cfg, background = parse_training_args(
        [
            "--config_path",
            str(config_file),
            "--policy.action_representation=alpha80",
            "--foreground",
        ]
    )
    assert background is False and cfg["data"]["rotation_representation"] == ROTATION
    specification.unlink()  # The resolved data configuration must contain all geometry.
    # Accelerate keeps precision in process-global state; isolate the BF16 trainer
    # from the other policies' FP32 tests, just as independent CLI runs are isolated.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import json, shutil",
                    "from pathlib import Path",
                    "from alohamini.learning.train_config import parse_training_args",
                    "from alohamini.learning.train import train",
                    "from alohamini.learning.checkpoint import read_checkpoint",
                    "cfg = parse_training_args()[0]",
                    "checkpoint = train(cfg)",
                    "path, _ = read_checkpoint(checkpoint)",
                    "saved = json.loads((path / 'train_config.json').read_text())",
                    "shutil.rmtree(cfg['model']['assets_path'])",
                    "shutil.rmtree(cfg['pretrained_path'])",
                    # Resume step 1 within the original two-step LR schedule.
                    "shutil.rmtree(path.parent)",
                    "saved.update(resume=True, _checkpoint=str(path.parent.parent / '000001'))",
                    "train(saved)",
                ]
            ),
            "--config_path",
            str(config_file),
            "--foreground",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoint = tmp_path / "run/checkpoint"
    loaded = NativePolicy(checkpoint, offline=True)
    assert loaded.config.action_representation == "alpha80"
    assert loaded.config.alpha80_layout == options["alpha80_layout"]
    assert loaded.manifest["data_fields"] == settings["data"]
    assert loaded.manifest["training"]["steps"] == 2
