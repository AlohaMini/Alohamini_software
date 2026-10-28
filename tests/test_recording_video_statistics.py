"""Encoding reuses RGB pixels without changing the pre-encoding statistics."""

import json
from collections import Counter
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from alohamini.datasets import images, video
from alohamini.datasets.image_statistics import sample_indices, sample_recording_images
from alohamini.datasets.record import dataset_schema


def episode_fixture(root, extension, shape, *, length=137, cameras=("forward", "wrist")):
    """Create only the recoverable input accepted by finalize_recording_video."""
    root.mkdir()
    features = {
        key: {"dtype": "float32", "shape": [1], "names": ["joint"]}
        for key in ("observation.state", "action")
    }
    height, width, _ = shape
    y, x = np.indices((height, width))
    rows = []
    for index in range(length):
        row = {
            "observation.state": [index / 10],
            "action": [-index / 10],
            "index": index,
            "episode_index": 0,
            "frame_index": index,
            "task_index": 0,
            "timestamp": index / 30,
            "task": "test",
        }
        for camera_index, camera in enumerate(cameras):
            relative = f"images/{camera}/frame_{index:06d}.{extension}"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.stack(
                (
                    (x * 17 + index * 13 + camera_index * 31) % 256,
                    (y * 23 + index * 29 + camera_index * 7) % 256,
                    ((x + y) * 11 + index * 37 + camera_index * 19) % 256,
                ),
                axis=-1,
            ).astype(np.uint8)
            if extension == "jpg":
                ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 70])
                assert ok
                path.write_bytes(encoded.tobytes())
            else:
                Image.fromarray(pixels).save(path, format="PNG")
            row[f"observation.images.{camera}"] = relative
        rows.append(row)
    image_format = "host-jpeg" if extension == "jpg" else "png"
    schema = dataset_schema(features, cameras, image_format)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "frames.parquet")
    (root / "episode.json").write_text(
        json.dumps({"length": length, "image_shapes": {camera: shape for camera in cameras}})
    )
    return features


def source_snapshot(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in [root / "frames.parquet", *sorted((root / "images").rglob("*"))]
        if path.is_file()
    }


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize(
    "extension,shape,length",
    [
        ("jpg", (18, 24, 3), 137),
        ("jpg", (17, 23, 3), 137),
        ("png", (17, 23, 3), 137),
        ("png", (302, 304, 3), 7),
    ],
)
def test_one_decode_per_frame_matches_old_sampled_statistics(
    tmp_path, monkeypatch, workers, extension, shape, length
):
    episode = tmp_path / "episode"
    cameras = ("forward", "wrist")
    features = episode_fixture(episode, extension, shape, length=length, cameras=cameras)
    # Keep the former independent pass as the oracle; do not derive expected
    # histograms from the encoder's collector or the post-encoding MP4 pixels.
    expected = sample_recording_images(episode, cameras, length).payload()
    references = pq.read_table(episode / "frames.parquet").to_pylist()
    expected_calls = Counter(
        (camera, row[f"observation.images.{camera}"]) for row in references for camera in cameras
    )
    calls, lock = Counter(), Lock()
    image_rgb = images.image_rgb

    def counted(root, camera, reference):
        with lock:
            calls[camera, reference] += 1
        return image_rgb(root, camera, reference)

    # Monitor both aliases so accidentally retaining the old sampling pass is
    # also counted, rather than watching only the encoder's own calls.
    monkeypatch.setattr(images, "image_rgb", counted)
    monkeypatch.setattr(video, "image_rgb", counted)
    video.finalize_recording_video(episode, 30, features, cameras, workers=workers)
    assert calls == expected_calls
    actual = json.loads((episode / "image_stats.json").read_text())
    assert actual == expected
    for camera in cameras:
        key = f"observation.images.{camera}"
        assert actual["cameras"][key]["frames"] == len(set(sample_indices(length)))
        assert video.inspect_video(episode / "videos" / f"{camera}.mp4", decode=True) == {
            "frames": length,
            "fps": 30.0,
            "shape": list(shape),
        }
    after = pq.read_table(episode / "frames.parquet").to_pylist()
    for old, current in zip(references, after, strict=True):
        for key in features:
            assert current[key] == old[key]
        for camera in cameras:
            assert current[f"observation.images.{camera}"]["frame_index"] == old["frame_index"]


@pytest.mark.parametrize("workers", [None, 2])
def test_no_cameras_keeps_empty_statistics_and_numeric_rows(tmp_path, monkeypatch, workers):
    episode = tmp_path / "episode"
    features = episode_fixture(episode, "jpg", (17, 23, 3), length=5, cameras=())
    before = pq.read_table(episode / "frames.parquet")
    expected = sample_recording_images(episode, (), 5).payload()

    def forbidden(*args, **kwargs):
        pytest.fail("No cameras must not decode or encode an image")

    monkeypatch.setattr(images, "image_rgb", forbidden)
    monkeypatch.setattr(video, "image_rgb", forbidden)
    monkeypatch.setattr(video, "_encode_frames", forbidden)
    video.finalize_recording_video(episode, 30, features, (), workers=workers)
    assert json.loads((episode / "image_stats.json").read_text()) == expected
    assert pq.read_table(episode / "frames.parquet").equals(before)
    assert not list((episode / "videos").glob("*.mp4"))


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("extension", ["jpg", "png"])
def test_bad_unsampled_frame_keeps_source_and_publishes_no_partial_statistics(
    tmp_path, workers, extension
):
    episode = tmp_path / "episode"
    cameras = ("forward", "wrist")
    features = episode_fixture(episode, extension, (17, 23, 3), cameras=cameras)
    bad_index = next(index for index in range(137) if index not in set(sample_indices(137)))
    bad = episode / "images" / "wrist" / f"frame_{bad_index:06d}.{extension}"
    bad.write_bytes(b"\xff\xd8not-a-jpeg\xff\xd9" if extension == "jpg" else b"bad PNG")
    before = source_snapshot(episode)
    # This corruption is invisible to sampling alone, but must still be caught
    # by the full encoder input traversal.
    sample_recording_images(episode, cameras, 137)
    with pytest.raises((OSError, ValueError)):
        video.finalize_recording_video(episode, 30, features, cameras, workers=workers)
    assert not (episode / "image_stats.json").exists()
    assert source_snapshot(episode) == before
    assert video.inspect_video(episode / "videos" / "forward.mp4")["frames"] == 137


@pytest.mark.parametrize("failure", ["header", "hash"])
def test_media_failure_keeps_original_index_and_does_not_publish_statistics(
    tmp_path, monkeypatch, failure
):
    episode = tmp_path / "episode"
    cameras = ("forward", "wrist")
    features = episode_fixture(episode, "jpg", (17, 23, 3), length=5, cameras=cameras)
    before = source_snapshot(episode)
    if failure == "header":
        inspect_video = video.inspect_video

        def invalid(path, **kwargs):
            actual = inspect_video(path, **kwargs)
            return {**actual, "frames": 0} if Path(path).stem == "wrist" else actual

        monkeypatch.setattr(video, "inspect_video", invalid)
        error, message = ValueError, "Encoded video does not match recording"
    else:
        file_sha256 = video.file_sha256

        def invalid_hash(path):
            if Path(path).stem == "wrist":
                raise OSError("injected video hash failure")
            return file_sha256(path)

        monkeypatch.setattr(video, "file_sha256", invalid_hash)
        error, message = OSError, "injected video hash failure"
    with pytest.raises(error, match=message):
        video.finalize_recording_video(episode, 30, features, cameras, workers=2)
    assert not (episode / "image_stats.json").exists()
    assert source_snapshot(episode) == before


def test_statistics_replace_failure_keeps_original_images_and_parquet(tmp_path, monkeypatch):
    from alohamini.datasets import record

    episode = tmp_path / "episode"
    features = episode_fixture(episode, "jpg", (17, 23, 3), length=5)
    before, replace = source_snapshot(episode), record.os.replace

    def fail_statistics(source, target):
        if Path(target).name == "image_stats.json":
            raise OSError("injected image statistics publication failure")
        return replace(source, target)

    monkeypatch.setattr(record.os, "replace", fail_statistics)
    with pytest.raises(OSError, match="injected image statistics"):
        video.finalize_recording_video(episode, 30, features, ("forward", "wrist"), workers=2)
    assert not (episode / "image_stats.json").exists()
    assert source_snapshot(episode) == before
