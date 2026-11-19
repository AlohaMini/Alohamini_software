"""桌面采集按钮映射到已有录制事件，结束后调用原生数据检查。"""

import logging
import sys
from contextlib import redirect_stdout
from queue import Empty

from alohamini.apps.recording import RecordingKeyboard, record
from alohamini.errors import AlohaMiniError
from alohamini.interaction import DesktopInteraction


class RecordingControls(RecordingKeyboard):
    def __init__(self, interaction, *, keyboard_enabled=False):
        super().__init__()
        self.interaction = interaction
        self.keyboard_enabled = keyboard_enabled
        self.phase = "starting"
        self.progress = {}
        self.token = None

    def __enter__(self):
        if self.keyboard_enabled:
            return super().__enter__()
        return self

    def begin(self, episode, total, saved):
        self.interaction.sequence += 1
        self.token = self.interaction.sequence
        self.progress = dict(episode=episode, total_episodes=total, saved_episodes=saved,
                             frames=0, remaining_s=0)
        self.report("recording")

    def report(self, phase, **values):
        self.phase = phase
        self.progress.update(values)
        self.interaction.notify(phase, progress=self.progress,
                                prompt_id=self.token if phase in ("recording", "resetting") else None)

    def countdown(self, remaining_s, frames):
        self.report(self.phase, remaining_s=remaining_s, frames=frames)

    def read(self):
        while True:
            try:
                command = self.interaction.responses.get_nowait()
            except Empty:
                break
            if isinstance(command, ValueError):
                raise command
            if command["prompt_id"] != self.token or self.phase not in ("recording", "resetting"):
                logging.warning("这次示范的操作已结束，请按当前页面状态操作。")
                continue
            self.update("right" if command["action"] == "end_episode" else "r", True)
        if self.interaction.cancelled.is_set():
            self.events.update(stop_recording=True, exit_early=True)
            return None
        if self.keyboard_enabled:
            return super().read()
        return set()


def check_recorded_dataset(path, *, dataset_name, task):
    from alohamini.datasets.tools import check_dataset

    checked = check_dataset(path, decode_images=True, decode_videos=True)
    summary = checked["summary"]
    episodes = summary.get("actual_episode_metadata_rows", summary.get("episodes", 0))
    frames = summary.get("actual_data_rows", summary.get("frames", 0))
    issues = checked["issues"]
    if not episodes or not frames:
        issues = [*issues, dict(severity="error", code="EMPTY_DATASET", message="没有可用于训练的示范，请重新采集。")]
    return dict(dataset_name=dataset_name, dataset_root=str(path), task=task,
                valid=checked["valid"] and episodes > 0 and frames > 0,
                episodes=episodes, frames=frames, issues=issues,
                errors=sum(issue["severity"] == "error" for issue in issues),
                warnings=sum(issue["severity"] == "warning" for issue in issues))


def run_gui_recording(**options):
    interaction = DesktopInteraction(sys.stdin, sys.stdout, actions=("end_episode", "rerecord", "finish"))
    with redirect_stdout(sys.stderr):
        try:
            interaction.notify("starting")
            path = record(**options, interaction=interaction)
            interaction.notify("checking_data")
            report = check_recorded_dataset(path, dataset_name=options["dataset_name"], task=options["task"])
        except (OSError, ValueError, RuntimeError, AlohaMiniError) as error:
            print(str(error), file=sys.stderr, flush=True)
            interaction.notify("failed")
            return 1
    interaction.notify("succeeded" if report["valid"] else "failed", report=report)
    return 0 if report["valid"] else 1
