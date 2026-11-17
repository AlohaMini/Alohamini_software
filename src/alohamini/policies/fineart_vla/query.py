"""Explicit native subtask and camera-bound VQA queries for FineART."""

import json
import re

from alohamini.datasets.recipe import DEFAULT_BINDINGS, render_message_turns


def parse_spatial(text):
    if "<loc" not in text:
        try:
            return json.loads(text)
        except ValueError:
            return None
    answers = []
    for part in text.split(" ; "):
        match = re.fullmatch(r"\s*(.+?)\s+((?:<loc\d{4}>\s*){2,4})\s*", part)
        if match is None:
            raise ValueError("FineART generated malformed spatial locations")
        coordinates = [int(value) for value in re.findall(r"<loc(\d{4})>", match[2])]
        if any(value > 1023 for value in coordinates):
            raise ValueError("FineART generated out-of-range spatial locations")
        answers.append((match[1].strip(), [value / 1023 for value in coordinates]))
    if len(answers) == 1 and len(answers[0][1]) == 2:
        label, (y, x) = answers[0]
        return dict(label=label, point_format="xy", point=[x, y])
    detections = []
    for label, coordinates in answers:
        if len(coordinates) != 4:
            raise ValueError("FineART spatial answer mixes boxes and points")
        y1, x1, y2, x2 = coordinates
        if x1 > x2 or y1 > y2:
            raise ValueError("FineART generated unordered spatial box corners")
        detections.append(dict(label=label, bbox_format="xyxy", bbox=[x1, y1, x2, y2]))
    return dict(detections=detections)


def query_language(model, processor, batch, *, kind="subtask", question=None, camera=None):
    if kind not in ("subtask", "vqa"):
        raise ValueError("FineART language query kind must be subtask or vqa")
    tasks = batch.get("task", [None] * len(batch["observation.state"]))
    messages = []
    if kind == "subtask":
        turns = processor.text.recipe.prompt_turns("subtask")
        for task in tasks:
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Subtask generation requires a nonempty high-level task")
            bindings = {name: None for name in DEFAULT_BINDINGS}
            bindings["task"] = task
            messages.append(render_message_turns(turns, bindings)["messages_rendered"])
    else:
        if not isinstance(camera, str) or not isinstance(question, str) or not question.strip():
            raise ValueError("VQA requires a selected camera and a nonempty question")
        camera = (
            camera if camera.startswith("observation.images.") else f"observation.images.{camera}"
        )
        if camera not in model.config.image_features:
            raise ValueError("VQA camera is not selected by this checkpoint")
        messages = [[dict(role="user", content=f"Image: {camera}\n{question}")] for _ in tasks]
    raw = {key: value for key, value in batch.items() if key not in ("action", "action_is_pad")}
    prepared = processor({**raw, "_language_query": messages})
    tokenizer = processor.text.tokenizer
    tokens = model.generate_text_tokens(prepared, eos_id=tokenizer.eos_token_id)
    results = []
    for sequence in tokens:
        text = tokenizer.decode(sequence, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("FineART generated an empty language response")
        result = dict(kind=kind, text=text)
        if kind == "vqa":
            result.update(camera=camera, answer=parse_spatial(text))
        results.append(result)
    return results
