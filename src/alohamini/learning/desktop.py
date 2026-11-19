"""桌面训练只承接交互与进度，训练和断点保存复用原生训练器。"""

import contextlib
import json
from pathlib import Path
import sys
import time

from alohamini.interaction import DesktopInteraction


class TrainingInteraction(DesktopInteraction):
    def __init__(self, source, output):
        super().__init__(source, output, actions=("finish",))
        self.path = None
        self.last_emit = 0
        self.started = time.monotonic()
        self.initial_step = None
        self.progress = dict(step=0, total_steps=0, loss=None, elapsed_s=0,
                             remaining_s=None, checkpoint="", saved_step=0, output_dir="", dataset="")

    def preparing(self, output, cfg):
        self.path = output / "studio-progress.json"
        self.progress.update(total_steps=cfg["steps"], output_dir=str(output), dataset=cfg["dataset"])
        if cfg.get("resume"):
            self.progress["checkpoint"] = str(Path(cfg["_checkpoint"]) / "pretrained_model")
            from alohamini.learning.checkpoint import read_checkpoint

            _, manifest = read_checkpoint(self.progress["checkpoint"])
            self.progress["saved_step"] = manifest["training"]["steps"]
        self.publish("preparing_training")

    def update(self, step, total, *, loss=None):
        if self.initial_step is None:
            self.initial_step = step
            self.started = time.monotonic()
        elapsed = time.monotonic() - self.started
        self.progress.update(step=step, total_steps=total, elapsed_s=int(elapsed),
                             remaining_s=int(elapsed / (step - self.initial_step) * (total - step))
                             if step > self.initial_step else None)
        if loss is not None:
            self.progress["loss"] = float(loss)
        if time.monotonic() - self.last_emit >= 1 or step == total:
            self.publish("training")

    def saving(self):
        self.publish("saving_checkpoint")

    def saved(self, checkpoint, step):
        self.progress.update(checkpoint=str(checkpoint), saved_step=step)
        self.publish("training")

    def complete(self, status):
        self.publish(status)

    def publish(self, step):
        self.last_emit = time.monotonic()
        if self.path is not None:
            temporary = self.path.with_suffix(".pending")
            temporary.write_text(json.dumps(dict(status=step, **self.progress), ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.path)
        self.notify(step, training=self.progress)


def training_runs(root):
    """只列出完整的原生断点；不加载权重或优化器。"""
    from alohamini.learning.checkpoint import read_checkpoint

    reports = []
    for path in sorted(Path(root).glob("*/checkpoint/train_config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            checkpoint = path.parent.resolve()
            if not (checkpoint.parent / "training_state/state.pt").is_file():
                continue
            _, manifest = read_checkpoint(checkpoint)
            cfg = json.loads(path.read_text(encoding="utf-8"))
            step = manifest["training"]["steps"]
            reports.append(dict(output_dir=cfg["output_dir"], dataset=cfg["dataset"],
                                checkpoint=str(checkpoint), saved_step=step, total_steps=cfg["steps"],
                                resumable=step < cfg["steps"]))
        except (OSError, ValueError, KeyError, TypeError):
            # 损坏或未写完的断点仍由既有模型检查报告，不作为续训入口。
            continue
    return reports


def main():
    interaction = TrainingInteraction(sys.stdin, sys.stdout)
    interaction.notify("preparing_training", training=interaction.progress)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from alohamini.environment import runtime_environment

            with runtime_environment(["train", *sys.argv[1:]]):
                from alohamini.learning.train_config import parse_training_args
                from alohamini.learning.train import train

                cfg, _ = parse_training_args()
                train(cfg, interaction=interaction)
    except (Exception, SystemExit) as error:
        print(f"训练未完成：{error}", file=sys.stderr, flush=True)
        interaction.complete("failed")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
