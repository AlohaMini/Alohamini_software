import copy
import json
import unittest
from unittest.mock import patch

from alohamini.errors import ProtocolError
from alohamini.protocol import HostSnapshot
from alohamini.remote.protocol import (
    context_for,
    decode_message,
    encode_message,
    snapshot_body,
    snapshot_from,
)

JPEG = b"\xff\xd8\x00original encoded image\xff\xd9"


def snapshot():
    return HostSnapshot(
        {
            "arm_left_shoulder_pan.pos": 2.0,
            "lift_axis.reference_sequence": 3,
            "_safety": {"host_session_id": "host-1", "control_epoch": 4},
            "_robot_metadata": {
                "robot_model": "alohamini2pro",
                "schema_version": 1,
                "axes": {"joint": {"unit": "degree", "offset": 0}},
            },
        },
        {"front": JPEG},
        100.0,
        100.02,
    )


class RemoteProtocolTests(unittest.TestCase):
    def test_snapshot_round_trip_preserves_jpeg_and_pi_clock_with_independent_state(self):
        source = snapshot()
        body = {"id": 12, "snapshot": snapshot_body(source), "task": "搬运"}
        parts = encode_message("observation", body, source.images)
        self.assertIs(parts[1], JPEG)
        kind, decoded, images = decode_message(parts)
        restored = snapshot_from(decoded["snapshot"], images)
        self.assertEqual(kind, "observation")
        self.assertEqual(restored.payload, source.payload)
        self.assertEqual(restored.images, source.images)
        self.assertEqual((restored.request_started_s, restored.received_s), (100.0, 100.02))
        self.assertIsNone(restored._command_context)
        restored.payload["_robot_metadata"]["axes"]["joint"]["offset"] = 20
        restored.images.clear()
        self.assertEqual(source.payload["_robot_metadata"]["axes"]["joint"]["offset"], 0)
        self.assertEqual(source.images, {"front": JPEG})

    def test_context_hash_is_canonical_and_tracks_only_control_frame_and_metadata(self):
        source = snapshot()
        original = context_for(source)
        self.assertEqual(original["host_session_id"], "host-1")
        self.assertEqual(original["control_epoch"], 4)
        self.assertEqual(original["lift_reference_sequence"], 3)
        self.assertEqual(len(original["robot_metadata_sha256"]), 64)
        source.payload["_robot_metadata"] = dict(
            reversed(list(source.payload["_robot_metadata"].items()))
        )
        source.payload["arm_left_shoulder_pan.pos"] = 200
        source.payload["_safety"]["feedback_valid"] = False
        self.assertEqual(context_for(source), original)
        for change in (
            lambda s: s.payload["_safety"].update(host_session_id="host-2"),
            lambda s: s.payload["_safety"].update(control_epoch=5),
            lambda s: s.payload.update({"lift_axis.reference_sequence": 4}),
            lambda s: s.payload["_robot_metadata"]["axes"]["joint"].update(offset=1),
        ):
            changed = snapshot()
            change(changed)
            self.assertNotEqual(context_for(changed), original)

    def test_rejects_ambiguous_and_nonfinite_json_on_both_paths(self):
        for raw in (
            b'{"version":1,"version":1,"type":"ping","body":{},"images":[]}',
            b'{"version":1,"type":"ping","body":{"a":1,"a":2},"images":[]}',
            b'{"version":1,"type":"ping","body":{"a":NaN},"images":[]}',
            b'{"version":1,"type":"ping","body":{"a":Infinity},"images":[]}',
            b'{"version":1,"type":"ping","body":{"a":1e999},"images":[]}',
            b"\xff",
        ):
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                decode_message([raw])
        for body in (
            {"a": float("nan")},
            {"a": [float("inf")]},
            {1: "non-string key"},
            {"a": b"no pickle bytes in JSON"},
            {"a": (1, 2)},
        ):
            with self.subTest(body=body), self.assertRaises(ProtocolError):
                encode_message("ping", body)

    def test_rejects_envelope_changes_frame_mismatch_and_invalid_images(self):
        header = json.loads(encode_message("observation", {}, {"front": JPEG})[0])
        for change in (
            {"version": True},
            {"version": 2},
            {"type": "pickle"},
            {"type": []},
            {"body": []},
            {"extra": "unsupported"},
            {"images": ["front", "front"]},
            {"images": ["../front"]},
            {"images": ["front\n"]},
            {"images": ["x" * 65]},
        ):
            with self.subTest(change=change), self.assertRaises(ProtocolError):
                decode_message([json.dumps({**header, **change}).encode(), JPEG])
        for parts in ([], [json.dumps(header).encode()], encode_message("ping", {}) + [JPEG]):
            with self.subTest(parts=parts), self.assertRaises(ProtocolError):
                decode_message(parts)
        for image in (b"", b"not jpeg", b"\xff\xd8truncated", bytearray(JPEG)):
            with self.subTest(image=image), self.assertRaises(ProtocolError):
                encode_message("observation", {}, {"front": image})
            with self.assertRaises(ProtocolError):
                decode_message([json.dumps(header).encode(), image])

    def test_frame_json_message_and_camera_limits_apply_on_encode_and_decode(self):
        parts = encode_message("observation", {}, {"front": JPEG})
        for constant, limit in (
            ("MAX_FRAME_BYTES", len(JPEG) - 1),
            ("MAX_JSON_BYTES", len(parts[0]) - 1),
            ("MAX_MESSAGE_BYTES", sum(map(len, parts)) - 1),
        ):
            with (
                self.subTest(constant=constant),
                patch(f"alohamini.remote.protocol.{constant}", limit),
            ):
                with self.assertRaises(ProtocolError):
                    encode_message("observation", {}, {"front": JPEG})
                with self.assertRaises(ProtocolError):
                    decode_message(parts)
        with self.assertRaises(ProtocolError):
            encode_message("observation", {}, {f"camera{i}": JPEG for i in range(17)})
        with self.assertRaises(ProtocolError):
            decode_message([parts[0]] + [JPEG] * 17)

    def test_snapshot_rejects_invalid_clock_values_and_context_fields(self):
        source = snapshot()
        body = snapshot_body(source)
        for values in ((True, 101), (-1, 101), (101, 100), (100, float("inf"))):
            bad = {**body, "request_started_s": values[0], "received_s": values[1]}
            with self.subTest(values=values), self.assertRaises(ProtocolError):
                snapshot_from(bad, source.images)
        for patch_safety in (
            {"host_session_id": ""},
            {"control_epoch": True},
            {"control_epoch": -1},
        ):
            changed = copy.deepcopy(source)
            changed.payload["_safety"].update(patch_safety)
            with self.subTest(patch=patch_safety), self.assertRaises(ProtocolError):
                context_for(changed)


if __name__ == "__main__":
    unittest.main()
