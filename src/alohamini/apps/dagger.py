"""Human-gated corrections using the platform's Host and dataset interfaces."""

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from alohamini._validation import finite_number
from alohamini.apps.teleoperation import KeyboardInput
from alohamini.calibration.encoder import HostPositionUnits


@dataclass(frozen=True)
class DAggerConfig:
    alignment_tolerance_deg: float = 3.0
    gripper_tolerance: float = 3.0
    settle_time_s: float = 0.3
    hold_output_limit: float = 0.2
    alignment_timeout_s: float = 60.0
    max_joint_velocity_deg_s: float = 60.0
    leader_id: str | None = None
    calibration_dir: str | None = None
    left_port: str = "/dev/am_arm_leader_left"
    right_port: str = "/dev/am_arm_leader_right"

    def __post_init__(self):
        for key in (
            "alignment_tolerance_deg",
            "gripper_tolerance",
            "settle_time_s",
            "hold_output_limit",
            "alignment_timeout_s",
            "max_joint_velocity_deg_s",
        ):
            value = getattr(self, key)
            finite_number(value, key)
            if value <= 0:
                raise ValueError(f"{key} must be positive")
        if self.hold_output_limit > 1:
            raise ValueError("hold_output_limit must be in (0, 1]")
        if self.alignment_tolerance_deg > 10 or self.gripper_tolerance > 10:
            raise ValueError("Alignment tolerances must not exceed 10")

    def leader_options(self):
        return {
            key: getattr(self, key)
            for key in ("leader_id", "calibration_dir", "left_port", "right_port")
        }


def add_dagger_arguments(parser):
    parser.add_argument("--dagger", action="store_true", help="暂停、主臂手动对齐与纠错采集")
    for name, help_text in (
        ("alignment_tolerance_deg", "关节对齐误差，默认 3 度"),
        ("gripper_tolerance", "夹爪对齐误差，默认 3 个开度百分点"),
        ("settle_time_s", "关节稳定后使能保持，默认 0.3 秒"),
        ("hold_output_limit", "主臂保持输出上限比例，默认 0.2"),
        ("alignment_timeout_s", "对齐及等待接管超时，默认 60 秒"),
        ("max_joint_velocity_deg_s", "人工关节目标变化速度上限，默认 60 度/秒"),
    ):
        import argparse

        parser.add_argument(
            f"--dagger.{name}", type=float, default=argparse.SUPPRESS, help=help_text
        )
    for flag, name in (
        ("--teleop.id", "leader_id"),
        ("--calibration_dir", "calibration_dir"),
        ("--left_port", "left_port"),
        ("--right_port", "right_port"),
    ):
        parser.add_argument(flag, dest=f"dagger.{name}", default=argparse.SUPPRESS)


def pop_dagger_options(options):
    enabled = options.pop("dagger", False)
    settings = {
        key.removeprefix("dagger."): options.pop(key)
        for key in list(options)
        if key.startswith("dagger.")
    }
    if settings and not enabled:
        raise ValueError("DAgger options require --dagger")
    return DAggerConfig(**settings) if enabled else None


class DAggerKeyboard(KeyboardInput):
    """Edge-triggered controls; held keys never advance multiple phases."""

    def __init__(self):
        super().__init__()
        self._events = deque()

    def update(self, name, pressed):
        with self._lock:
            if name in ("esc", "q") and pressed:
                self._quit = True
            if name not in ("space", "tab"):
                return
            if pressed and name not in self._pressed:
                # Keep at most one operator request pending. A sequence typed
                # while inference blocks must not skip the alignment gate.
                if not self._events:
                    self._events.append(name)
            if pressed:
                self._pressed.add(name)
            else:
                self._pressed.discard(name)

    def events(self):
        if self.read() is None:
            return ["quit"]
        with self._lock:
            result = list(self._events)
            self._events.clear()
            return result


class ManualHandover:
    """Policy-independent pause/alignment/correction state machine.

    The leader is never driven to the follower. Its aligned joints are held at
    their own measured positions until the operator explicitly starts correction.
    Only correction segments are recorded, each as a separate dataset episode.
    """

    def __init__(self, leader, keyboard, config=None):
        self.leader, self.keyboard = leader, keyboard
        self.config = config or DAggerConfig()
        self.phase = "policy"
        self.stopped = False
        self.recording = False
        self.save_pending = False
        self._settled = {}
        self._held = set()
        self._previous_leader = None
        self._previous_time = None
        self._hold = None
        self._last_action = None
        self._first_human = False
        self._alignment_started = None
        self._last_notice = None
        self._save_future = self._save_executor = None

    def _notice(self, message):
        if message != self._last_notice:
            logging.warning("[DAgger] %s", message)
            self._last_notice = message

    @staticmethod
    def hold_targets(snapshot):
        result = {
            f"{name}.pos": snapshot.payload[f"{name}.pos"]
            for name in snapshot.payload["_robot_metadata"]["motors"]
            if name.startswith("arm_")
        }
        result.update(
            {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "lift_axis.height_mm": snapshot.payload["lift_axis.height_mm"],
            }
        )
        return result

    @staticmethod
    def _scale(name, snapshot):
        """Physical degrees per Host wire unit, with gripper opening kept in %."""
        motor = snapshot.payload["_robot_metadata"]["motors"][name.removesuffix(".pos")]
        if name.endswith("_gripper.pos"):
            if motor["normalization"] != "range_0_100":
                raise ValueError("DAgger grippers require range_0_100 units")
            return 1.0
        if motor["normalization"] == "degrees":
            return 1.0
        units = HostPositionUnits(
            **{key: motor[key] for key in ("normalization", "range_min", "range_max", "drive_mode")}
        )
        return (units.range_max - units.range_min) * 360 / 4096 / 200

    def _pause(self, snapshot, policy, dataset):
        # Clearing the inference generation invalidates outstanding cloud/RTC
        # results as well as local ACT/chunk history.
        policy.reset()
        self.phase = "paused"
        self._hold = self.hold_targets(snapshot)
        self._last_action = dict(self._hold)
        self.leader.release_holds()
        if self.recording:
            dataset.event({"type": "sequence_boundary", "reason": "human_correction_end"})
            self.recording = False
            self.save_pending = True
        self._notice("已暂停；Tab 对齐主臂，Space 恢复策略，Esc 退出。")

    def handle_events(self, snapshot, policy, dataset, *, now, refresh=None):
        self.finish_save(dataset)
        events = self.keyboard.events()
        if events and refresh is not None:
            fresh = refresh()
            if fresh is None:
                self.feedback_lost(snapshot, policy, dataset)
                return
            snapshot = fresh
        for event in events:
            if event == "quit":
                self.stopped = True
                return
            if event == "space":
                if self.phase == "policy":
                    self._pause(snapshot, policy, dataset)
                elif self.phase in ("aligning", "ready"):
                    self._pause(snapshot, policy, dataset)
                elif self.phase == "paused" and not self.save_pending:
                    self.leader.release_holds()
                    policy.reset()
                    self.phase = "policy"
                    self._notice("策略恢复；Space 暂停。")
                elif self.phase == "paused":
                    self._notice("正在保存纠错；保存完成后按 Space 交还策略。")
            elif event == "tab":
                if self.phase == "paused" and not self.save_pending:
                    self.phase = "aligning"
                    self._held.clear()
                    self._settled.clear()
                    self._previous_leader = self._previous_time = None
                    self._alignment_started = now
                    self._notice("手动对齐主臂，关节到位后逐个保持；从臂保持不动。")
                elif self.phase == "ready":
                    # Check every joint again immediately before release.
                    pose = self.leader.read_handover(self._units(snapshot))
                    if not self._aligned(pose, snapshot):
                        self._pause(snapshot, policy, dataset)
                        self._notice("主臂位置已偏移；Tab 重新对齐。")
                        continue
                    self.leader.release_holds()
                    self.phase = "human"
                    self._first_human = True
                    self._previous_time = now
                    if dataset is not None:
                        dataset.begin_episode()
                        dataset.event(
                            {"type": "sequence_boundary", "reason": "human_correction_start"}
                        )
                        self.recording = True
                    self._notice("人工纠错中；Tab 结束并保存。")
                elif self.phase == "human":
                    self._pause(snapshot, policy, dataset)
                elif self.phase in ("aligning", "ready"):
                    self._pause(snapshot, policy, dataset)

    @staticmethod
    def _units(snapshot):
        return {
            f"{name}.pos": motor["normalization"]
            for name, motor in snapshot.payload["_robot_metadata"]["motors"].items()
            if name.startswith("arm_")
        }

    def _tolerance(self, name):
        return (
            self.config.gripper_tolerance
            if name.endswith("_gripper.pos")
            else self.config.alignment_tolerance_deg
        )

    def _aligned(self, pose, snapshot):
        return pose and all(
            abs(pose[key] - snapshot.payload[key]) * self._scale(key, snapshot)
            <= self._tolerance(key)
            for key in pose
        )

    def action(self, snapshot, *, now):
        if self.phase == "policy":
            return None
        if self.phase == "paused":
            if self._hold is None:
                self._hold = self.hold_targets(snapshot)
                self._last_action = dict(self._hold)
            return dict(self._hold)
        if (
            self._alignment_started is not None
            and now - self._alignment_started >= self.config.alignment_timeout_s
        ):
            # Only the alignment/ready phases are time-limited; correction can
            # last as long as the evaluation session.
            if self.phase in ("aligning", "ready"):
                raise TimeoutError("DAgger alignment timed out; leader holds released on exit")
        pose = self.leader.read_handover(self._units(snapshot))
        if not pose or set(pose) != set(self._units(snapshot)):
            raise ValueError("DAgger requires every configured leader joint")
        if self.phase in ("aligning", "ready"):
            previous = self._previous_leader
            dt = None if self._previous_time is None else now - self._previous_time
            for key in pose:
                scale = self._scale(key, snapshot)
                close = abs(pose[key] - snapshot.payload[key]) * scale <= self._tolerance(key)
                slow = (
                    previous is not None
                    and dt is not None
                    and dt > 0
                    and abs(pose[key] - previous[key]) * scale / dt <= 5.0
                )
                if key in self._held and not close:
                    raise RuntimeError("Aligned leader/follower positions diverged")
                if key in self._held:
                    continue
                if close and slow:
                    self._settled.setdefault(key, now)
                else:
                    self._settled.pop(key, None)
            eligible = [
                key
                for key, started in self._settled.items()
                if key not in self._held and now - started >= self.config.settle_time_s
            ]
            if eligible:
                key = eligible[0]  # Bound acknowledged serial work per control tick.
                self.leader.hold_joint(key, output_limit=self.config.hold_output_limit)
                self._held.add(key)
                self._notice(f"已对齐并保持 {len(self._held)}/{len(pose)}：{key}")
            if len(self._held) == len(pose):
                self.phase = "ready"
                self._notice("全部到位；扶稳主臂后按 Tab 释放保持并开始纠错。")
            self._previous_leader, self._previous_time = pose, now
            return dict(self._hold)
        # Releasing torque can let a supported arm settle. Recheck before the
        # first command, then bound all subsequent manual target changes.
        if self._first_human:
            if not self._aligned(pose, snapshot):
                raise RuntimeError("Leader moved during release; pause and realign")
            self._first_human = False
            self._previous_time = now
            return dict(self._hold)
        dt = max(0.0, min(now - self._previous_time, 0.1))
        action = dict(self._hold)
        for key, target in pose.items():
            scale = self._scale(key, snapshot)
            speed = 50.0 if key.endswith("_gripper.pos") else self.config.max_joint_velocity_deg_s
            step = speed * dt / scale
            old = self._last_action[key]
            action[key] = old + max(-step, min(step, target - old))
        self._last_action, self._previous_time = action, now
        return action

    def feedback_lost(self, snapshot, policy, dataset):
        """Never keep latched leaders across a watchdog or missing robot feedback."""
        self._pause(snapshot, policy, dataset)
        self._hold = None  # Relatch measured positions only after fresh feedback returns.

    def poll_disconnected(self):
        # A disconnected session may exit, but cannot queue a future takeover.
        if "quit" in self.keyboard.events():
            self.stopped = True

    def finish_save(self, dataset):
        if self.save_pending:
            if self._save_future is None:
                self._save_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="DAggerSave"
                )
                self._save_future = self._save_executor.submit(dataset.save_episode)
            elif self._save_future.done():
                self._save_future.result()
                self._save_executor.shutdown()
                self._save_future = self._save_executor = None
                self.save_pending = False
                self._notice("纠错已保存；Tab 再次对齐，Space 交还策略。")

    def close(self):
        try:
            self.leader.release_holds()
        finally:
            if self._save_executor is not None:
                self._save_executor.shutdown(wait=True)
                self._save_future.result()
