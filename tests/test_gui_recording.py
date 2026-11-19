"""图形采集复用真实数据写入和检查；只替换机器人、主臂与实时采样边界。"""

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow.parquet as pq
import pytest
from test_dataset import frame, jpeg
from test_recording_storage import metadata
from test_teleoperation import snapshot

from alohamini.apps import recording
from alohamini.apps.recording_interaction import RecordingControls, check_recorded_dataset
from alohamini.interaction import DesktopInteraction


def test_preview_uses_recorded_rgb_and_throttles_without_new_camera_requests():
    import base64
    from PIL import Image

    with gui_channel(lambda *_: None) as (interaction, events):
        controls = RecordingControls(interaction)
        controls.begin(1, 2, 0)
        frame = snapshot()
        frame.images["forward"] = jpeg()
        controls.preview(frame, None)
        controls.preview(frame, None)
        previews = [event for event in events if "previews" in event]
        assert len(previews) == 1
        image = previews[0]["previews"][0]
        with Image.open(io.BytesIO(base64.b64decode(image["base64"]))) as decoded:
            assert decoded.mode == "RGB" and decoded.width <= 480 and decoded.height <= 270


@contextlib.contextmanager
def gui_channel(on_event):
    read_fd, write_fd = os.pipe()
    events = []
    with os.fdopen(read_fd) as source, os.fdopen(write_fd, "w"):
        class Output(io.StringIO):
            def flush(output):
                event = json.loads(output.getvalue().splitlines()[-1])
                events.append(event)
                on_event(interaction, event)

        interaction = DesktopInteraction(source, Output(), actions=("end_episode", "rerecord", "finish"))
        yield interaction, events


@contextlib.contextmanager
def recording_hardware(capture):
    client = MagicMock(client_id="client")
    initial = snapshot()
    initial.payload["_robot_metadata"] = metadata()
    client.connect_control.return_value = initial
    client.__enter__.return_value = client
    with patch.object(recording, "HostClient", return_value=client), patch.object(
        recording, "BimanualLeader", return_value=MagicMock()
    ), patch.object(recording, "record_loop", side_effect=capture):
        yield


def add_frames(dataset, marker):
    for _ in range(3):
        values = frame(dataset)
        values["action"][:] = marker
        assert dataset.add_frame(values, {"forward": jpeg()}, {})


def test_end_and_rerecord_use_existing_events_and_keep_only_accepted_attempt(tmp_path):
    def buttons(interaction, event):
        if event["step"] != "recording":
            return
        if event["prompt_id"] == 1:
            interaction.responses.put(dict(action="rerecord", prompt_id=1))
            return
        interaction.responses.put(dict(action="end_episode", prompt_id=2))

    markers = []

    def capture(_client, _model, _leader, keyboard, *, dataset=None, **_options):
        if dataset is not None:
            markers.append(len(markers) + 1)
            add_frames(dataset, markers[-1])
        keyboard.read()
        if len(markers) == 2 and dataset is None:
            keyboard.interaction.cancelled.set()
            keyboard.read()

    with gui_channel(buttons) as (interaction, events), recording_hardware(capture), patch.object(
        recording.RecordingKeyboard, "__enter__", side_effect=AssertionError("主臂模式不应请求键盘权限")
    ):
        root = recording.record("fake", "alohamini2pro", dataset_name="task", task="拿积木",
            root=tmp_path / "task", num_episodes=2, reset_time_s=1,
            interaction=interaction, keyboard_enabled=False)
    report = check_recorded_dataset(root, dataset_name="task", task="拿积木")
    assert report["valid"]
    assert (report["episodes"], report["frames"]) == (1, 3)
    rows = [row for path in root.glob("data/**/*.parquet") for row in pq.read_table(path).to_pylist()]
    assert all(row["action"][0] == 2 for row in rows)
    assert any(event["step"] == "episode_discarded" for event in events)
    assert any(event["step"] == "episode_saved" for event in events)


def test_finish_during_rejected_attempt_never_saves_it_and_empty_data_cannot_train(tmp_path):
    def buttons(interaction, event):
        if event["step"] == "recording":
            interaction.responses.put(dict(action="rerecord", prompt_id=event["prompt_id"]))
        if event["step"] == "resetting":
            interaction.cancelled.set()

    def capture(_client, _model, _leader, keyboard, *, dataset=None, **_options):
        if dataset is not None:
            add_frames(dataset, 9)
        keyboard.read()

    with gui_channel(buttons) as (interaction, _), recording_hardware(capture):
        root = recording.record("fake", "alohamini2pro", dataset_name="task", task="拿积木",
            root=tmp_path / "task", interaction=interaction, keyboard_enabled=False)
    report = check_recorded_dataset(root, dataset_name="task", task="拿积木")
    assert not report["valid"]
    assert report["episodes"] == 0
    assert any(issue["code"] == "EMPTY_DATASET" for issue in report["issues"])


def test_stale_buttons_do_not_end_the_next_attempt_or_resume_capture():
    with gui_channel(lambda *_: None) as (interaction, _):
        controls = RecordingControls(interaction)
        controls.begin(1, 2, 0)
        controls.begin(1, 2, 0)
        interaction.responses.put(dict(action="rerecord", prompt_id=1))
        controls.read()
        assert not controls.events["exit_early"]
        assert not controls.events["rerecord_episode"]


@pytest.mark.parametrize("corrupt,natural", [(False, False), (True, False), (False, True)])
def test_cli_buttons_finish_then_automatically_decode_and_check_data(tmp_path, corrupt, natural):
    script = """
import time,sys
from unittest.mock import patch
from alohamini.cli import main
from alohamini.apps import recording_interaction
from test_gui_recording import recording_hardware,add_frames
original_check=recording_interaction.check_recorded_dataset
def check(path,**options):
    if sys.argv[2] == 'true': next(path.glob('videos/**/*.mp4')).unlink()
    return original_check(path,**options)
def capture(client,model,leader,keyboard,*,dataset=None,**options):
    if dataset is not None: add_frames(dataset, 3)
    if sys.argv[3] == 'true': return
    deadline=time.monotonic()+4
    while not keyboard.events['exit_early']:
        keyboard.read()
        if time.monotonic()>deadline: raise RuntimeError('桌面回复超时')
        time.sleep(0.005)
with recording_hardware(capture), patch.object(recording_interaction,'check_recorded_dataset',side_effect=check):
    code=main(['record','--gui','--host=fake','--robot_model=alohamini2pro',
        '--dataset=task','--task=拿积木','--root='+sys.argv[1],
        '--num_episodes='+('1' if sys.argv[3]=='true' else '2'),'--reset_time=0'])
sys.exit(code)
"""
    root = tmp_path / "task"
    with subprocess.Popen([sys.executable, "-u", "-c", script, str(root), str(corrupt).lower(), str(natural).lower()],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(Path("src").resolve()), str(Path("tests").resolve())])},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8") as child:
        events = []
        for line in child.stdout:
            event = json.loads(line)
            events.append(event)
            if event["step"] == "recording" and not natural:
                child.stdin.write('{"action":"finish"}\n')
                child.stdin.flush()
        child.wait(timeout=8)
        error = child.stderr.read()
    result = events[-1]
    if not corrupt:
        assert child.returncode == 0, error
        assert any(event["step"] == "checking_data" for event in events)
        assert result["step"] == "succeeded"
        assert result["report"]["dataset_root"] == str(root)
        assert (result["report"]["episodes"], result["report"]["frames"]) == (1, 3)
    if corrupt:
        assert child.returncode != 0, error
        assert result["step"] == "failed"
