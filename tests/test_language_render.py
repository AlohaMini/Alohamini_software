"""Recorded-time language routing and supervision, independent of model frameworks."""

import json

import pytest

from alohamini.datasets.language_render import (
    active_at,
    emitted_at,
    nth_prev,
    render_sample,
)
from alohamini.datasets.recipe import TrainingRecipe


def subtask(text, t):
    return dict(role="assistant", style="subtask", content=text, timestamp=t)


def recipe(**kwargs):
    return TrainingRecipe.from_dict(kwargs)


def test_persistent_annotations_are_causal_and_events_use_recorded_time():
    rows = [subtask("approach", 0.0), subtask("grasp", 0.1)]
    assert active_at(0.09, persistent=rows, style="subtask")["content"] == "approach"
    assert emitted_at(0.09, persistent=rows, events=[], style="subtask") is None
    assert emitted_at(0.1, persistent=rows, events=[], style="subtask")["content"] == "grasp"
    assert nth_prev(0.09, persistent=rows, style="subtask") is None
    with pytest.raises(ValueError, match="positive integer"):
        nth_prev(0.09, persistent=rows, style="subtask", offset=-1)


def test_high_level_supervision_does_not_become_low_level_action_supervision():
    high = recipe(
        messages=[
            dict(role="user", content="${task}", stream="high_level"),
            dict(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )
    assert render_sample(recipe=high, persistent=[], events=[], t=0, sample_idx=0) is None
    rendered = render_sample(
        recipe=high, persistent=[subtask("grasp", 0)], events=[], t=0, sample_idx=0, task="pick"
    )
    assert rendered["messages_rendered"][1]["content"] == "grasp"
    assert rendered["target_message_indices"] == [1]
    assert "low_level" not in rendered["message_streams"]


def test_camera_specific_vqa_routes_only_when_recorded_and_never_guesses_view():
    camera = "observation.images.wrist_left"
    frames = [
        dict(role="assistant", content="blue", style="vqa", camera=camera),
        dict(role="assistant", content="red", style="vqa", camera="observation.images.forward"),
    ]
    with pytest.raises(ValueError, match="Ambiguous resolver"):
        emitted_at(0, persistent=[], events=frames, style="vqa", role="assistant")
    task = dict(role="user", content="${task}", stream="low_level")
    blend = recipe(
        blend={
            "action": dict(weight=1, messages=[task]),
            "question": dict(
                weight=1,
                route="vqa",
                bindings={
                    "vqa": f"emitted_at(t, style=vqa, role=assistant, camera={camera})",
                },
                messages=[
                    dict(
                        role="assistant",
                        content="${vqa}",
                        stream="high_level",
                        target=True,
                        if_present="vqa",
                    )
                ],
            ),
        }
    )
    rendered = render_sample(recipe=blend, persistent=[], events=frames, t=0, sample_idx=3)
    assert rendered["messages_rendered"][0]["content"] == "blue"
    ordinary = render_sample(recipe=blend, persistent=[], events=[], t=0, sample_idx=3, task="pick")
    assert ordinary["message_streams"] == ["low_level"]


def test_blend_is_deterministic_and_task_rephrasing_cannot_read_future_rows():
    blend = recipe(
        blend={
            name: dict(
                weight=weight,
                messages=[dict(role="user", content=name + " ${task}", stream="low_level")],
            )
            for name, weight in [("a", 0.3), ("b", 0.7)]
        }
    )
    rows = [dict(role="user", style="task_aug", content="future", timestamp=1)]
    expected = [
        render_sample(
            recipe=blend, persistent=rows, events=[], t=0, sample_idx=i, default_task="recorded"
        )
        for i in range(100)
    ]
    repeated = {
        i: render_sample(
            recipe=blend, persistent=rows, events=[], t=0, sample_idx=i, default_task="recorded"
        )
        for i in reversed(range(100))
    }
    assert all(expected[i] == repeated[i] for i in range(100))
    assert all("future" not in str(item) for item in expected)
    assert {item["messages_rendered"][0]["content"] for item in expected} == {
        "a recorded",
        "b recorded",
    }


def test_json_tool_calls_render_and_recipes_remain_portable():
    source = dict(
        messages=[
            dict(
                role="assistant",
                content=None,
                stream="high_level",
                target=True,
                if_present="speech",
                tool_calls_from="speech",
            )
        ]
    )
    portable = json.loads(json.dumps(source))
    rows = [
        dict(
            role="assistant",
            style="subtask",
            content=None,
            timestamp=0,
            tool_calls=[json.dumps(dict(function=dict(name="say", arguments="hello")))],
        )
    ]
    rows[0].pop("timestamp")
    rows[0]["style"] = None
    rendered = render_sample(
        recipe=recipe(**portable), persistent=[], events=rows, t=0, sample_idx=0
    )
    assert rendered["messages_rendered"][0]["tool_calls"][0]["function"]["name"] == "say"
    for weight in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="positive weight"):
            recipe(
                blend={
                    "bad": dict(
                        weight=weight,
                        messages=[dict(role="user", content="task", stream="low_level")],
                    )
                }
            )
