"""Language control cadence, task boundaries and failure behavior."""

import pytest

from alohamini.learning.language_control import LanguageController


def test_subtasks_refresh_by_observation_time_and_reset_on_new_task():
    controller = LanguageController(2.0)
    calls = []
    answers = iter(["approach", "approach", "grasp", "release"])

    def query(task):
        calls.append(task)
        return next(answers)

    assert controller.condition("pick", 10.0, query) == ("approach", True)
    assert controller.condition("pick", 11.99, query) == ("approach", False)
    assert controller.condition("pick", 12.0, query) == ("approach", False)
    assert controller.condition("pick", 14.0, query) == ("grasp", True)
    assert controller.generation == 2
    assert controller.condition("place", 14.1, query) == ("release", True)
    assert calls == ["pick", "pick", "pick", "place"]
    controller.reset()
    assert controller.subtask is None and controller.generation == 0


def test_invalid_planner_output_does_not_commit_controller_state():
    controller = LanguageController(1)
    for output in ("", "   ", None):
        with pytest.raises(ValueError, match="invalid subtask"):
            controller.condition("pick", 0.0, lambda _: output)
        assert controller.subtask is None and controller.last_query_s is None
    controller.condition("pick", 2.0, lambda _: "grasp")
    with pytest.raises(ValueError, match="nondecreasing"):
        controller.condition("pick", 1.0, lambda _: "release")
    assert controller.subtask == "grasp"


@pytest.mark.parametrize("interval", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_controller_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        LanguageController(interval)
