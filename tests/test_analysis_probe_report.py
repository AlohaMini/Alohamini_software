"""Probe reports preserve per-action units and compare against persistence."""

from copy import deepcopy
from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np

from alohamini.analysis.report import export_html


class Tables(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def report(tmp_path, probe=None, *, files=()):
    output = tmp_path / "report"
    output.mkdir()
    for name in files:
        (output / name).write_bytes(b"existing probe artifact")
    summary = {
        "dataset": "/data/evaluation",
        "episode": 0,
        "segment": 0,
        "time_basis": "host_state_monotonic",
        "selected_rows": 2,
        "duration_s": 0.1,
        "recorded_frame_hz": 10,
        "fixed_dimensions": None,
        "chunk_assumption": None,
        "warnings": [],
        "interpretation": [],
        "control_diagnostics": False,
        "model_context": {"checkpoint": {"kind": "act"}},
    }
    if probe is not None:
        summary["demonstration_probe"] = probe
    analysis = SimpleNamespace(
        data=SimpleNamespace(root=tmp_path / "evaluation", media={}),
        summary=summary,
        series={},
        reference=None,
        events=[],
        times=np.array([0, 0.1]),
        anchor_s=0,
    )
    export_html(analysis, output, reader=None)
    return (output / "report.html").read_text()


def test_probe_shows_algorithm_windows_named_metrics_baseline_and_artifacts(tmp_path):
    probe = {
        "schema_version": 2,
        "kind": "smolvla",
        "samples": 3,
        "split": "train",
        "episode": 7,
        "action_names": ["lift_axis.height_mm", "arm_left_joint.pos"],
        "action_units": ["mm", "range_m100_100"],
        "window": {
            "predicted_steps": 50,
            "executed_steps": 10,
            "fps": 30,
            "action_offsets": list(range(-1, 49)),
            "observation_offsets": [-1, 0],
        },
        "artifacts": {
            "predictions_npz": "demonstration_predictions.npz",
            "probe_json": "demonstration_probe.json",
        },
        "excluded_fixed_dimensions": ["arm_right_joint.pos"],
        "interpretation": "示范观测上的离线预测，不代表闭环成功率。",
        "notes": ["缺少同名状态时，保持当前观测值的基线不可用。", "模型不会读取未来训练图像。"],
        "metrics": {
            "arm_left_joint.pos": {
                "unit": "range_m100_100",
                "first_action_mae": 0,
                "chunk_mae": 1.234567,
                "persistence_first_action_mae": 2.5,
                "persistence_chunk_mae": 3.75,
                "valid_action_steps": 120,
            },
            "lift_axis.height_mm": {
                "unit": "mm",
                "first_action_mae": 4.5,
                "chunk_mae": 6.75,
                "persistence_first_action_mae": 8,
                "persistence_chunk_mae": 9.5,
                "valid_action_steps": 119,
            },
        },
    }
    before = deepcopy(probe)
    page = report(tmp_path, probe, files=probe["artifacts"].values())
    assert "算法：smolvla" in page
    assert "示范划分：train" in page and "Episode：7" in page and "采样观测数：3" in page
    assert "预测窗口（步）：50" in page and "执行窗口（步）：10" in page
    assert "数据集帧率（Hz）：30" in page
    assert "历史前缀与无效尾部不计入" in page
    assert "不跨单位合并" in page and "不代表闭环成功率" in page
    assert "<li>缺少同名状态时，保持当前观测值的基线不可用。</li>" in page
    assert "<li>模型不会读取未来训练图像。</li>" in page
    assert "不计入误差的固定维度：arm_right_joint.pos" in page
    rows = Tables(page).rows
    lift = rows.index(["lift_axis.height_mm", "mm", "4.5", "6.75", "8", "9.5", "119"])
    joint = rows.index(["arm_left_joint.pos", "range_m100_100", "0", "1.235", "2.5", "3.75", "120"])
    assert lift < joint  # Use the declared action order rather than dictionary insertion order.
    assert '<a href="demonstration_predictions.npz">预测与标签 NPZ</a>' in page
    assert '<a href="demonstration_probe.json">探针结果 JSON</a>' in page
    assert probe == before


def test_legacy_probe_retains_metrics_and_marks_unavailable_values_without_broken_links(tmp_path):
    page = report(
        tmp_path,
        {
            "samples": 1,
            "split": "val",
            "episode": 0,
            "interpretation": "训练集误差不代表泛化。",
            "metrics": {
                "joint<&>": {
                    "unit": "rad",
                    "first_action_mae": None,
                    "chunk_mae": None,
                    "valid_action_steps": 0,
                }
            },
        },
        files=("demonstration_predictions.npz",),
    )
    assert "算法：act" in page
    assert ["joint<&>", "rad", "不可用", "不可用", "不可用", "不可用", "0"] in Tables(page).rows
    assert "joint&lt;&amp;&gt;" in page
    assert 'href="demonstration_predictions.npz"' in page
    assert 'href="demonstration_probe.json"' not in page
    assert "预测窗口（步）" not in page


def test_report_without_probe_does_not_add_probe_section_or_downloads(tmp_path):
    page = report(tmp_path)
    assert "评估动作与参考示范" in page
    assert "模型在示范上的预测误差" not in page
    assert "demonstration_predictions.npz" not in page
    assert "demonstration_probe.json" not in page


def test_probe_distinguishes_model_and_persistence_valid_step_counts(tmp_path):
    page = report(
        tmp_path,
        {
            "samples": 2,
            "split": "train",
            "episode": 0,
            "metrics": {
                "joint.pos": {
                    "unit": "rad",
                    "first_action_mae": 1,
                    "chunk_mae": 2,
                    "persistence_first_action_mae": 3,
                    "persistence_chunk_mae": 4,
                    "valid_action_steps": 50,
                    "persistence_valid_action_steps": 31,
                }
            },
        },
    )
    assert "有效步数（模型 / 基线）" in page
    assert ["joint.pos", "rad", "1", "2", "3", "4", "模型50 / 基线31"] in Tables(page).rows
