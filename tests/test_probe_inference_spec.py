"""Offline probes use causal FastWAM inputs without changing video supervision."""

from copy import deepcopy

import pytest
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.policies.registry import algorithm


@pytest.mark.parametrize(
    "options,action_offsets,video_offsets",
    [
        ({}, list(range(32)), list(range(0, 33, 4))),
        (
            {"action_horizon": 7, "num_video_frames": 9, "action_video_freq_ratio": 2},
            list(range(7)),
            [0, 2, 4, 6, 8],
        ),
    ],
)
def test_fastwam_inference_keeps_action_contract_and_training_video_windows(
    options, action_offsets, video_offsets
):
    recipe = algorithm("fastwam")
    original = deepcopy(options)
    cameras = ["left", "right"]
    training = recipe.sample_spec(options, cameras=cameras)

    assert recipe.inference_spec(options, cameras=cameras) == {
        "delta_indices": {"action": action_offsets},
        "include_task": True,
    }
    assert training == {
        "delta_indices": {
            "action": action_offsets,
            "observation.images.left": video_offsets,
            "observation.images.right": video_offsets,
        },
        "include_task": True,
    }
    assert recipe.sample_spec(options, cameras=cameras) == training
    assert options == original


@pytest.mark.parametrize("method", ["sample_spec", "inference_spec"])
def test_fastwam_requires_cameras_for_both_sample_contracts(method):
    with pytest.raises(ValueError, match="requires recorded cameras"):
        getattr(algorithm("fastwam"), method)({})


def test_fastwam_probe_reads_only_current_camera_frame(recording, monkeypatch):
    recipe = algorithm("fastwam")
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.inference_spec({"action_horizon": 3}, cameras=["forward"]),
    )
    reads = []
    original = data._value

    def read(index, key):
        reads.append((index, key))
        return original(index, key)

    monkeypatch.setattr(data, "_value", read)
    sample = data[1]

    assert [(index, key) for index, key in reads if key.startswith("observation.images.")] == [
        (data.sample_indices[1], "observation.images.forward")
    ]
    assert sample["observation.images.forward"].shape == (3, 32, 32)
    assert "observation.images.forward_is_pad" not in sample
    assert sample["observation.state"].ndim == 1
    assert sample["action"].shape[0] == 3
    assert sample["action_is_pad"].tolist() == [False, False, False]
    assert sample["task"] == "test"
