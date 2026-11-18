"""LeRobot v3 language rows, without a LeRobot or Hugging Face datasets runtime.

Schema: LeRobot datasets/language.py at7910c20a75834ad3fa6a9792df4d44a614d1bba6.
Persistent rows carry emission timestamps; event rows belong to their exact frame.
"""

import json
import math

LANGUAGE_COLUMNS = ("language_persistent", "language_events")
PERSISTENT_STYLES = {"subtask", "plan", "memory", "motion", "task_aug"}
EVENT_STYLES = {"interjection", "vqa", "trace"}
VIEW_DEPENDENT_STYLES = {"vqa", "trace"}


def column_for_style(style):
    if style is None:
        return "language_events"
    if style in PERSISTENT_STYLES:
        return "language_persistent"
    if style in EVENT_STYLES:
        return "language_events"
    raise ValueError(f"Unknown language style: {style!r}")


def language_feature_info():
    return {key: dict(dtype="language", shape=[1], names=None) for key in LANGUAGE_COLUMNS}


def language_arrow_type(*, persistent):
    import pyarrow as pa

    fields = [
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string()),
        pa.field("style", pa.string()),
    ]
    if persistent:
        fields.append(pa.field("timestamp", pa.float32(), nullable=False))
    fields.extend([pa.field("camera", pa.string()), pa.field("tool_calls", pa.list_(pa.string()))])
    return pa.list_(pa.struct(fields))


def validate_camera_field(style, camera):
    if style in VIEW_DEPENDENT_STYLES:
        if not isinstance(camera, str) or not camera.startswith("observation.images."):
            raise ValueError("View-dependent language requires a recorded camera feature")
    elif camera is not None:
        raise ValueError("Camera-independent language styles must not bind a camera")


def validate_language_rows(rows, *, persistent, cameras=None):
    if rows is None:
        return
    if not isinstance(rows, list):
        raise ValueError("Language annotations must be a list of rows or null")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("role") not in (
            "user",
            "assistant",
            "system",
            "tool",
        ):
            raise ValueError(f"Language row {index}: invalid role")
        if row.get("content") is not None and not isinstance(row["content"], str):
            raise ValueError(f"Language row {index}: content must be text or null")
        style = row.get("style")
        if style is not None:
            expected = "language_persistent" if persistent else "language_events"
            if column_for_style(style) != expected:
                raise ValueError(f"Language row {index}: style belongs in another column")
        validate_camera_field(style, row.get("camera"))
        if cameras is not None and row.get("camera") is not None and row["camera"] not in cameras:
            raise ValueError(f"Language row {index}: camera is not recorded")
        if persistent:
            timestamp = row.get("timestamp")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (float, int))
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise ValueError(f"Language row {index}: invalid persistent timestamp")
        calls = row.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ValueError(f"Language row {index}: tool_calls must be a list or null")
            for call in calls:
                if isinstance(call, str):
                    try:
                        call = json.loads(call)
                    except ValueError as exc:
                        raise ValueError(f"Language row {index}: invalid tool-call JSON") from exc
                if not isinstance(call, dict):
                    raise ValueError(f"Language row {index}: tool call must be an object")
