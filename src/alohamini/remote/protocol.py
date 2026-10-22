"""Bounded JSON/JPEG messages for a Pi policy client and a remote server.

JPEG payloads remain encoded throughout this module. Authentication and transport
encryption belong to the surrounding VPN or SSH tunnel.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from alohamini.errors import ProtocolError
from alohamini.protocol import MAX_FRAME_BYTES, MAX_JSON_BYTES, MAX_MESSAGE_BYTES, HostSnapshot

VERSION = 1
MAX_CAMERAS = 16
MAX_MESSAGE_FRAMES = 1 + MAX_CAMERAS
KINDS = frozenset(
    {"hello", "ready", "observation", "actions", "error", "ping", "pong", "stop", "ack"}
)
_CAMERA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _check_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ProtocolError("Remote JSON nesting exceeds 64 levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("Remote JSON object keys must be strings")
            _check_json(item, depth + 1)
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item, depth + 1)
        return
    raise ProtocolError("Remote messages require finite JSON values")


def _json_bytes(value: Any, *, canonical: bool = False) -> bytes:
    _check_json(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=canonical,
            separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ProtocolError(f"Invalid remote JSON: {exc}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"Non-finite JSON number: {value}")


def _camera_names(names: Any) -> None:
    if not isinstance(names, list) or len(names) > MAX_CAMERAS:
        raise ProtocolError("Remote images must list at most 16 camera names")
    if any(not isinstance(name, str) or not _CAMERA_NAME.fullmatch(name) for name in names):
        raise ProtocolError("Invalid remote camera name")
    if len(set(names)) != len(names):
        raise ProtocolError("Duplicate remote camera names")


def _image_bytes(name: str, image: Any) -> None:
    if not isinstance(image, bytes) or not 4 <= len(image) <= MAX_FRAME_BYTES:
        raise ProtocolError(f"Invalid JPEG frame size or type for {name}")
    if not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
        raise ProtocolError(f"Remote image {name} must contain JPEG bytes")


def encode_message(kind: str, body: dict, images: dict[str, bytes] | None = None) -> list[bytes]:
    """Encode one strict JSON envelope followed by unchanged JPEG frames."""
    if not isinstance(kind, str) or kind not in KINDS:
        raise ProtocolError("Unsupported remote message type")
    if not isinstance(body, dict):
        raise ProtocolError("Remote message body must be an object")
    if images is None:
        images = {}
    if not isinstance(images, dict):
        raise ProtocolError("Remote images must be a camera-to-JPEG object")
    names = list(images)
    _camera_names(names)
    for name, image in images.items():
        _image_bytes(name, image)
    envelope = {"version": VERSION, "type": kind, "body": body, "images": names}
    header = _json_bytes(envelope)
    if len(header) > MAX_JSON_BYTES:
        raise ProtocolError("Remote JSON frame exceeds the size limit")
    parts = [header, *images.values()]
    if sum(map(len, parts)) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Remote message exceeds the multipart size limit")
    return parts


def decode_message(parts: list[bytes]) -> tuple[str, dict, dict[str, bytes]]:
    """Validate frame limits and a strict envelope without decoding images."""
    if not isinstance(parts, (list, tuple)) or not 1 <= len(parts) <= MAX_MESSAGE_FRAMES:
        raise ProtocolError("Invalid remote multipart frame count")
    if any(not isinstance(part, bytes) for part in parts):
        raise ProtocolError("Remote multipart frames must be bytes")
    if len(parts[0]) > MAX_JSON_BYTES or any(len(part) > MAX_FRAME_BYTES for part in parts):
        raise ProtocolError("Remote multipart frame exceeds the size limit")
    if sum(map(len, parts)) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Remote message exceeds the multipart size limit")
    try:
        envelope = json.loads(
            parts[0].decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError(f"Invalid remote JSON: {exc}") from exc
    _check_json(envelope)
    if not isinstance(envelope, dict) or set(envelope) != {"version", "type", "body", "images"}:
        raise ProtocolError("Invalid remote envelope fields")
    if type(envelope["version"]) is not int or envelope["version"] != VERSION:
        raise ProtocolError("Unsupported remote protocol version")
    kind, body, names = (envelope[key] for key in ("type", "body", "images"))
    if not isinstance(kind, str) or kind not in KINDS:
        raise ProtocolError("Unsupported remote message type")
    if not isinstance(body, dict):
        raise ProtocolError("Remote message body must be an object")
    _camera_names(names)
    if len(parts) != 1 + len(names):
        raise ProtocolError("Remote camera names and multipart frames disagree")
    images = dict(zip(names, parts[1:], strict=True))
    for name, image in images.items():
        _image_bytes(name, image)
    return kind, body, images


def _snapshot_fields(body: dict) -> tuple[dict, float, float]:
    if not isinstance(body, dict) or set(body) != {"payload", "request_started_s", "received_s"}:
        raise ProtocolError("Invalid remote snapshot fields")
    payload = body["payload"]
    if not isinstance(payload, dict):
        raise ProtocolError("Remote snapshot payload must be an object")
    started, received = body["request_started_s"], body["received_s"]
    try:
        valid = all(
            type(value) in (int, float) and math.isfinite(value) for value in (started, received)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ProtocolError("Remote snapshot timestamps must be finite numbers")
    if not 0 <= started <= received:
        raise ProtocolError("Remote snapshot timestamps must be ordered and nonnegative")
    return payload, started, received


def snapshot_body(snapshot: HostSnapshot) -> dict:
    """Copy snapshot state and Pi-clock timing; JPEGs travel in separate frames."""
    if not isinstance(snapshot, HostSnapshot):
        raise ProtocolError("Expected a HostSnapshot")
    body = {
        "payload": snapshot.payload,
        "request_started_s": snapshot.request_started_s,
        "received_s": snapshot.received_s,
    }
    _snapshot_fields(body)
    return json.loads(_json_bytes(body))


def snapshot_from(body: dict, images: dict[str, bytes]) -> HostSnapshot:
    """Reconstruct a snapshot without interpreting Pi times on the server clock."""
    payload, started, received = _snapshot_fields(body)
    # Reuse the wire validation and independent JSON ownership for local callers.
    _, copied, copied_images = decode_message(encode_message("observation", payload, images))
    return HostSnapshot(copied, copied_images, started, received)


def context_for(snapshot: HostSnapshot) -> dict:
    """Identify the Host control frame and canonical robot configuration."""
    if not isinstance(snapshot, HostSnapshot):
        raise ProtocolError("Expected a HostSnapshot")
    payload = snapshot.payload
    if not isinstance(payload, dict):
        raise ProtocolError("Remote context requires a Host payload object")
    status, metadata = payload.get("_safety"), payload.get("_robot_metadata")
    if not isinstance(status, dict) or not isinstance(metadata, dict):
        raise ProtocolError("Remote context requires Host safety and robot metadata")
    session, epoch = status.get("host_session_id"), status.get("control_epoch")
    if not isinstance(session, str) or not session.strip() or len(session) > 64:
        raise ProtocolError("Remote context requires a valid Host session")
    if type(epoch) is not int or epoch < 0:
        raise ProtocolError("Remote context requires a nonnegative control epoch")
    reference = payload.get("lift_axis.reference_sequence")
    if reference is not None and (type(reference) is not int or reference < 0):
        raise ProtocolError("Invalid lift reference sequence")
    return {
        "host_session_id": session,
        "control_epoch": epoch,
        "lift_reference_sequence": reference,
        "robot_metadata_sha256": hashlib.sha256(_json_bytes(metadata, canonical=True)).hexdigest(),
    }
