"""FineART normalized-image VQA targets; label-first PaliGemma location tokens.

Protocol: official FineART processor at SOURCE.json. Unlike its permissive
clamping, invalid units/formats fail before they can become incorrect labels.
"""

import json
import math


def location(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Spatial VQA coordinates must be numbers")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Spatial VQA requires finite image fractions in [0,1], not pixels")
    return f"<loc{round(value * 1023):04d}>"


def spatial_answer(payload):
    if "point" in payload:
        point = payload["point"]
        if (
            payload.get("point_format") != "xy"
            or not isinstance(point, (list, tuple))
            or len(point) != 2
        ):
            raise ValueError("Spatial keypoints require point_format=xy and two coordinates")
        label = payload.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Spatial VQA targets require a nonempty label")
        return f"{label.strip()} {location(point[1])}{location(point[0])}"
    if "detections" not in payload:
        return None
    detections = payload["detections"]
    if not isinstance(detections, list) or not detections:
        raise ValueError("Spatial detections must be a nonempty list")
    targets = []
    for item in detections:
        box = item.get("bbox") if isinstance(item, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4 or item.get("bbox_format") != "xyxy":
            raise ValueError("Spatial boxes require bbox_format=xyxy and four coordinates")
        x1, y1, x2, y2 = box
        tokens = [location(value) for value in (y1, x1, y2, x2)]
        if x1 > x2 or y1 > y2:
            raise ValueError("Spatial box corners must be ordered")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Spatial VQA targets require a nonempty label")
        targets.append(f"{label.strip()} {''.join(tokens)}")
    return " ; ".join(targets)


def spatial_messages(item, *, cameras, fast_enabled, fast_skip_tokens):
    result = [dict(message) for message in item["messages_rendered"]]
    source_cameras = item.get("message_cameras", [[] for _ in result])
    for index in item["target_message_indices"]:
        content = result[index].get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        target = spatial_answer(payload) if isinstance(payload, dict) else None
        if target is None:
            continue
        views = source_cameras[index]
        if len(views) != 1 or views[0] not in cameras:
            raise ValueError("Spatial VQA targets must bind one selected recorded camera")
        if fast_enabled and fast_skip_tokens < 1152:
            raise ValueError(
                "Spatial VQA with FAST requires fast_skip_tokens>=1152 to reserve loc tokens"
            )
        result[index]["content"] = target
    return result
