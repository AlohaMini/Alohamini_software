"""Distinct front views survive video encoding, data selection and policy checkpoints."""

import json
from contextlib import closing

import numpy as np
import pytest
import torch
from test_dataset import frame, jpeg
from test_native_learning import model_options
from test_recording_storage import add_episode, create, metadata

from alohamini.datasets.record import LocalDataset
from alohamini.datasets.tools import check_dataset
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, make_policy, make_processor, save_checkpoint
from alohamini.learning.processor import act_statistics
from alohamini.model import get_robot_model


@pytest.mark.parametrize("background", [False, True])
def test_dual_front_video_to_act_checkpoint_preserves_view_identity(tmp_path, background):
    cameras = get_robot_model("alohamini2edu").default_cameras
    colors = dict(zip(cameras, [(230, 20, 20), (20, 230, 20), (20, 20, 230)], strict=True))
    pictures = {}
    for name, color in colors.items():
        # Native recording uses the deployed Host JPEG color convention;
        # ROS camera-stream JPEGs use a separate standard-RGB payload.
        pictures[name] = jpeg(color, (32, 32, 3))
    root = tmp_path / "recording"
    with closing(LocalDataset(
        root, fps=30, task="pick", robot_metadata=metadata(cameras),
        background_video_encoding=background, video_encoding_workers=2,
    )) as writer:
        for _ in range(2):
            writer.begin_episode()
            for index in range(4):
                values = frame(writer)
                values["action"][:] = index
                # Arrival order must not assign pictures to a different camera.
                assert writer.add_frame(values, dict(reversed(list(pictures.items()))),
                                        {"safety": {"feedback_valid": True}})
            writer.save_episode()
    report = check_dataset(root, decode_videos=True)
    assert report["valid"], report
    info = json.loads((root / "meta/info.json").read_text())
    assert info["total_frames"] == 8
    assert len(list((root / "videos").rglob("*.mp4"))) == 6
    samples = AlohaMiniDataset(root, episodes=[0, 1], chunk_size=3, image_size=(32, 32))
    assert tuple(samples.cameras) == cameras
    for index in (0, 3, 4, 7):
        for name, color in colors.items():
            pixels = samples[index][f"observation.images.{name}"]
            np.testing.assert_allclose(pixels.mean((1, 2)).numpy() * 255, color, atol=12)
    selected = AlohaMiniDataset(root, episodes=[0], cameras=list(reversed(cameras)))
    assert selected.cameras == list(reversed(cameras))

    options = model_options(use_camera_embeddings=True)
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    stats = act_statistics(samples)
    model = make_policy("act", options, stats)
    raw = {key: value[None] for key, value in samples[0].items() if torch.is_tensor(value)}
    processor = make_processor(model, stats, "cpu")
    loss, _ = model(processor(raw))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.model.encoder_camera_embed.weight.grad).all()
    checkpoint = tmp_path / "policy"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    loaded = NativePolicy(checkpoint, offline=True)
    assert tuple(loaded.manifest["cameras"]) == cameras
    expected_keys = [f"observation.images.{name}" for name in cameras]
    assert list(loaded.config.image_features) == expected_keys
    model.eval()
    with torch.inference_mode():
        expected = model.predict_action_chunk(processor(raw))
        actual = loaded.model.predict_action_chunk(
            loaded.processor(dict(reversed(list(raw.items()))))
        )
    torch.testing.assert_close(actual, expected)


def test_resume_preserves_legacy_camera_feature_order(tmp_path):
    cameras = get_robot_model("alohamini2edu").default_cameras
    root = create(tmp_path / "record", episodes=1, cameras=cameras)
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    # Historical files could inherit an arbitrary first-frame arrival order.
    info["features"] = dict(reversed(list(info["features"].items())))
    path.write_text(json.dumps(info))
    before = AlohaMiniDataset(root, episodes=[0]).cameras
    assert before == list(reversed(cameras))
    with closing(LocalDataset(
        root, fps=30, task="pick", robot_metadata=metadata(cameras), resume=True,
    )) as dataset:
        add_episode(dataset, 2)
        dataset.save_episode()
    assert AlohaMiniDataset(root, episodes=[0, 1]).cameras == before
