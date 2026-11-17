"""Optional language-conditioned action control, independent of policy architecture."""

import math


class LanguageController:
    """Refresh a subtask by observation time; an episode/task reset clears all state."""

    def __init__(self, interval_s):
        if (
            isinstance(interval_s, bool)
            or not isinstance(interval_s, (int, float))
            or not math.isfinite(interval_s)
            or interval_s <= 0
        ):
            raise ValueError("language_replan_interval_s must be finite and positive")
        self.interval_s = float(interval_s)
        self.reset()

    def reset(self):
        self.task = self.subtask = self.last_query_s = self.last_observation_s = None
        self.generation = 0

    def condition(self, task, observed_s, query):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Language control requires an explicit high-level task")
        if (
            isinstance(observed_s, bool)
            or not isinstance(observed_s, (int, float))
            or not math.isfinite(observed_s)
        ):
            raise ValueError("Language control requires a finite observation timestamp")
        task_changed = task != self.task
        if (
            not task_changed
            and self.last_observation_s is not None
            and observed_s < self.last_observation_s
        ):
            raise ValueError("Language-control observations must have nondecreasing timestamps")
        due = (
            task_changed
            or self.last_query_s is None
            or observed_s - self.last_query_s >= self.interval_s
        )
        changed = False
        if due:
            result = query(task)
            if not isinstance(result, str) or not result.strip() or len(result) > 4096:
                raise ValueError("Language planner returned an empty or invalid subtask")
            subtask = result.strip()
            changed = task_changed or subtask != self.subtask
            self.task, self.subtask, self.last_query_s = task, subtask, float(observed_s)
            if changed:
                self.generation += 1
        self.last_observation_s = float(observed_s)
        return self.subtask, changed
