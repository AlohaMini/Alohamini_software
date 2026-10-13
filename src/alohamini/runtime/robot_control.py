"""Whole-robot coordination over the existing arm and base/lift controllers."""

import time
from collections.abc import Callable

from alohamini.errors import CommandRejectedError
from alohamini.hardware.feedback import FeedbackBatch
from alohamini.runtime.arm_control import ArmController
from alohamini.runtime.base_lift_control import BaseLiftController
from alohamini.runtime.lifecycle import CommandSubmission
from alohamini.schema import BodyVelocity, CommandIdentity, RobotCommand


class RobotController:
    """Validate all requested subsystems before staging any of them.

    Device writes remain sequential, not physically atomic. A partial write failure
    propagates to HostSupervisor, which stops and disables every attempted bus.
    """

    def __init__(self, arms: ArmController, base_lift: BaseLiftController) -> None:
        self.arms = arms
        self.base_lift = base_lift
        self._session: str | None = None
        self.timing_ms: dict[str, float] = {}

    def bind_session(self, session: str) -> None:
        if self._session is not None:
            raise RuntimeError("Robot controller cannot be rebound")
        self._session = session
        self.arms.bind_session(session)
        self.base_lift.bind_session(session)

    def observe(self, batches: tuple[FeedbackBatch, ...]) -> None:
        self.timing_ms = {}
        self.arms.observe(batches)
        self.base_lift.observe(batches)

    def begin_lift_homing(self) -> None:
        self.base_lift.begin_lift_homing()
        self.arms.clear_targets()

    def _base_target(self, command: RobotCommand) -> BodyVelocity:
        if command.base_velocity is not None:
            return command.base_velocity
        previous = self.base_lift.base_target
        return BodyVelocity() if previous is None else previous.body_velocity

    def validate_targets(self, command: RobotCommand) -> None:
        self._prepare_write(command)

    def _prepare_write(self, command: RobotCommand) -> Callable[[], None]:
        """Validate all subsystems before returning a cycle-local staging callback."""
        started = time.perf_counter()
        if not isinstance(command, RobotCommand):
            raise TypeError("Expected RobotCommand")
        if self.base_lift.lift_homing:
            raise CommandRejectedError("Lift homing is in progress")
        stages = []
        try:
            if command.positions_rad:
                stages.append(self.arms._prepare_targets(command.positions_rad))
            if (
                command.base_velocity is not None
                or command.lift_height_m is not None
                or command.lift_stop
            ):
                stages.append(
                    self.base_lift._prepare_targets(
                        self._base_target(command),
                        lift_height_m=command.lift_height_m,
                        lift_stop=command.lift_stop,
                    )
                )
        except (ValueError, RuntimeError) as exc:
            raise CommandRejectedError(str(exc)) from exc
        prepared_s = time.perf_counter() - started

        def stage():
            started = time.perf_counter()
            for apply in stages:
                apply()
            self.timing_ms["prepare"] = (prepared_s + time.perf_counter() - started) * 1e3

        return stage

    def set_targets(self, command: RobotCommand) -> None:
        self._prepare_write(command)()

    def submission(self, identity: CommandIdentity, command: RobotCommand) -> CommandSubmission:
        return CommandSubmission(
            identity,
            lambda: self.set_targets(command),
            lambda: self._prepare_write(command),
        )

    def supervise(self) -> None:
        started = time.perf_counter()
        self.arms.supervise()
        arms_done = time.perf_counter()
        self.base_lift.supervise()
        self.timing_ms["arms"] = (arms_done - started) * 1e3
        self.timing_ms["base_lift"] = (time.perf_counter() - arms_done) * 1e3

    def clear_targets(self) -> None:
        try:
            self.arms.clear_targets()
        finally:
            self.base_lift.clear_targets()
