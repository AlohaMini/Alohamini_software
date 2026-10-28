"""Host JPEG fast decoding preserves the deployed OpenCV pixel contract."""

import io
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFile

from alohamini.datasets import images as wire


def pixels(pattern, shape):
    height, width = shape
    y, x = np.indices(shape)
    if pattern == "random":
        return np.random.default_rng(184).integers(0, 256, (height, width, 3), np.uint8)
    if pattern == "gradient":
        return np.stack(
            ((x * 13 + y * 7) % 256, (x * 3 + y * 19) % 256, (x * 23 + y * 5) % 256), axis=-1
        ).astype(np.uint8)
    colors = np.array([[240, 15, 30], [10, 20, 245], [30, 235, 15]], np.uint8)
    return colors[(x + y // 3) % 3]


def encode(array, *, quality=70, progressive=False, subsampling=2, **options):
    stream = io.BytesIO()
    Image.fromarray(array).save(
        stream,
        format="JPEG",
        quality=quality,
        progressive=progressive,
        subsampling=subsampling,
        **options,
    )
    return stream.getvalue()


def old_decode(encoded):
    """Independent copy of the old validation + OpenCV decode behavior."""
    if (
        not 4 <= len(encoded) <= 32 * 1024 * 1024
        or not encoded.startswith(b"\xff\xd8")
        or not encoded.endswith(b"\xff\xd9")
    ):
        raise ValueError("Invalid Host JPEG")
    with Image.open(io.BytesIO(encoded)) as image:
        if (
            image.format != "JPEG"
            or image.mode != "RGB"
            or image.width * image.height > 32 * 1024**2
        ):
            raise ValueError("Invalid image format, mode or dimensions")
        image.load()
        shape = (image.height, image.width, 3)
    result = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if result is None or result.shape != shape:
        raise ValueError("Invalid Host JPEG")
    return result


@pytest.mark.parametrize("pattern", ["colors", "random", "gradient"])
@pytest.mark.parametrize("shape", [(1, 1), (17, 29), (16, 24)])
@pytest.mark.parametrize(
    "quality,progressive,subsampling",
    [
        (5, False, 2),
        (70, False, 2),
        (100, False, 0),
        (95, True, 1),
        (30, True, 2),
        (80, False, 1),
    ],
)
def test_plain_host_pixels_match_old_decoder_without_a_second_jpeg_decode(
    pattern,
    shape,
    quality,
    progressive,
    subsampling,
):
    encoded = encode(
        pixels(pattern, shape), quality=quality, progressive=progressive, subsampling=subsampling
    )
    expected = old_decode(encoded)
    original_load, decoded = ImageFile.ImageFile.load, []

    def counted_load(image, *args, **kwargs):
        # Array/byte extraction may ask an already loaded image to load again;
        # count actual JPEG decoder entries, not these no-op accessor calls.
        if image.format == "JPEG" and image.tile:
            decoded.append(image.size)
        return original_load(image, *args, **kwargs)

    with (
        patch.object(ImageFile.ImageFile, "load", counted_load),
        patch("cv2.imdecode", side_effect=AssertionError("Ordinary JPEG decoded twice")),
    ):
        actual = wire.decode_host_image(encoded)
    assert decoded == [(shape[1], shape[0])]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous and actual.flags.writeable
    assert actual.flags.owndata
    unchanged = actual.copy()
    actual[:] = 0
    with patch("cv2.imdecode", side_effect=AssertionError("Ordinary JPEG decoded twice")):
        again = wire.decode_host_image(encoded)
    np.testing.assert_array_equal(again, unchanged)
    assert not np.shares_memory(actual, again)


def add_marker(encoded, marker, payload):
    return (
        encoded[:2]
        + b"\xff"
        + bytes([marker])
        + (len(payload) + 2).to_bytes(2, "big")
        + payload
        + encoded[2:]
    )


@pytest.mark.parametrize("pattern", ["colors", "random"])
def test_actual_host_opencv_encoding_keeps_deployed_rgb_interpretation(pattern):
    # Match camera.py: feed the captured RGB array directly to the BGR-oriented
    # JPEG encoder, then interpret the legacy imdecode result as RGB.
    frame = pixels(pattern, (17, 29))
    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    assert success
    encoded = buffer.tobytes()
    expected = old_decode(encoded)
    with patch("cv2.imdecode", side_effect=AssertionError("Host JPEG decoded twice")):
        actual = wire.decode_host_image(encoded)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous and actual.flags.writeable and actual.flags.owndata


def nonstandard_components(encoded):
    """Change SOF/SOS component identifiers without changing the JPEG scan."""
    data, offset, found_sof = bytearray(encoded), 2, False
    replacement = {1: 7, 2: 8, 3: 9}
    while offset < len(data):
        assert data[offset] == 255
        marker = data[offset + 1]
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            assert data[offset + 9] == 3
            for index in range(3):
                position = offset + 10 + index * 3
                data[position] = replacement[data[position]]
            found_sof = True
        if marker == 0xDA:
            assert data[offset + 4] == 3 and found_sof
            for index in range(3):
                position = offset + 5 + index * 2
                data[position] = replacement[data[position]]
            return bytes(data)
        offset += length + 2
    raise AssertionError("Fixture has no JPEG scan")


@pytest.mark.parametrize("metadata", ["icc", "adobe", "components"])
def test_unfamiliar_metadata_or_component_ids_keep_opencv_fallback(metadata):
    array = pixels("gradient", (19, 27))
    if metadata == "icc":
        encoded = encode(array, icc_profile=b"opaque ICC metadata; no new color conversion")
    elif metadata == "adobe":
        encoded = add_marker(encode(array), 0xEE, b"Adobe\x00\x64\x00\x00\x00\x00\x01")
    else:
        encoded = nonstandard_components(encode(array))
    expected = old_decode(encoded)
    with patch("cv2.imdecode", wraps=cv2.imdecode) as legacy:
        actual = wire.decode_host_image(encoded)
    assert legacy.call_count == 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("orientation", [2, 3, 6, 8])
@pytest.mark.parametrize("shape", [(17, 29), (17, 17)])
def test_exif_preserves_old_orientation_and_shape_rejection(orientation, shape):
    exif = Image.Exif()
    exif[274] = orientation
    encoded = encode(pixels("gradient", shape), exif=exif)
    with patch("cv2.imdecode", wraps=cv2.imdecode) as legacy:
        if orientation in (6, 8) and shape[0] != shape[1]:
            with pytest.raises(ValueError, match="Invalid Host JPEG"):
                old_decode(encoded)
            legacy.reset_mock()
            with pytest.raises(ValueError, match="Invalid Host JPEG"):
                wire.decode_host_image(encoded)
        else:
            expected = old_decode(encoded)
            legacy.reset_mock()
            np.testing.assert_array_equal(wire.decode_host_image(encoded), expected)
        assert legacy.call_count == 1


def test_internal_scan_damage_retaining_eoi_is_rejected_by_pillow_before_fallback():
    encoded = encode(pixels("random", (63, 65)), quality=95)
    marker = encoded.index(b"\xff\xda")
    scan = marker + 2 + int.from_bytes(encoded[marker + 2 : marker + 4], "big")
    # Preserve all shape headers and EOI, truncate the entropy scan and insert
    # an invalid marker so the strict decoder cannot silently fill missing rows.
    damaged = encoded[: scan + 7] + b"\xff\x02\xff\xd9"
    assert damaged.endswith(b"\xff\xd9")
    with Image.open(io.BytesIO(damaged)) as image:
        assert image.mode == "RGB" and image.size == (65, 63)
        with pytest.raises(OSError):
            image.load()
    with patch("cv2.imdecode", side_effect=AssertionError("Pillow-rejected image accepted")):
        with pytest.raises(ValueError, match="Invalid Host JPEG"):
            wire.decode_host_image(damaged)


@pytest.mark.parametrize("mode", ["L", "CMYK"])
def test_non_rgb_modes_remain_rejected_before_pixel_decode(mode):
    image = Image.new(mode, (17, 19))
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    with (
        patch.object(
            ImageFile.ImageFile, "load", side_effect=AssertionError("Invalid mode decoded")
        ),
        patch("cv2.imdecode", side_effect=AssertionError("Invalid mode decoded")),
    ):
        with pytest.raises(ValueError, match="Invalid image format, mode or dimensions"):
            wire.decode_host_image(stream.getvalue())


@pytest.mark.parametrize("limit", ["pixels", "bytes"])
def test_size_limits_still_reject_before_pixel_decode(monkeypatch, limit):
    encoded = encode(pixels("random", (17, 19)))
    monkeypatch.setattr(
        wire,
        "MAX_IMAGE_PIXELS" if limit == "pixels" else "MAX_IMAGE_BYTES",
        16 if limit == "pixels" else len(encoded) - 1,
    )
    with (
        patch.object(
            ImageFile.ImageFile, "load", side_effect=AssertionError("Oversized image decoded")
        ),
        patch("cv2.imdecode", side_effect=AssertionError("Oversized image decoded")),
    ):
        with pytest.raises(ValueError):
            wire.decode_host_image(encoded)


def test_prepare_recording_image_keeps_original_bytes_and_header_only_validation():
    encoded = encode(pixels("gradient", (17, 29)))
    with (
        patch.object(
            ImageFile.ImageFile, "load", side_effect=AssertionError("Capture loaded pixels")
        ),
        patch("cv2.imdecode", side_effect=AssertionError("Capture decoded JPEG")),
    ):
        prepared, shape = wire.prepare_recording_image(encoded)
    assert prepared is encoded and shape == (17, 29, 3)


def test_saved_mp4_preserves_physical_red_green_blue_from_host_wire(tmp_path):
    import contextlib

    import av
    from test_dataset import frame
    from test_recording_storage import metadata

    from alohamini.datasets.record import LocalDataset

    expected = np.zeros((96, 192, 3), np.uint8)
    for channel in range(3):
        expected[:, channel * 64 : (channel + 1) * 64, channel] = 255
    # Camera.read returns BGR; reproduce the Host's deployed encoding boundary.
    bgr = np.ascontiguousarray(expected[..., ::-1])
    host_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".jpg", host_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    root = tmp_path / "record"
    with contextlib.closing(
        LocalDataset(root, fps=30, task="colors", robot_metadata=metadata())
    ) as dataset:
        dataset.begin_episode()
        for _ in range(3):
            assert dataset.add_frame(frame(dataset), {"forward": encoded.tobytes()}, {})
        dataset.save_episode()
    video = next((root / "videos").rglob("*.mp4"))
    with av.open(str(video)) as container:
        saved = next(container.decode(video=0)).to_ndarray(format="rgb24")
    for channel in range(3):
        center = saved[16:80, channel * 64 + 16 : channel * 64 + 48]
        color = np.median(center, axis=(0, 1))
        assert color[channel] > 240
        assert np.max(np.delete(color, channel)) < 15
