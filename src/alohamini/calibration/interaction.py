"""桌面标定的逐行 JSON 交互；标定算法和事务仍由 procedure 管理。"""

import json
import sys
import threading
from contextlib import redirect_stdout
from queue import Empty, Queue


class CalibrationInteraction:
    def __init__(self, source, output):
        self.output = output
        self.responses = Queue()
        self.cancelled = threading.Event()
        self.sequence = 0
        threading.Thread(target=self._read, args=(source,), daemon=True).start()

    def _read(self, source):
        for line in source:
            try:
                response = json.loads(line)
                if not isinstance(response, dict) or response.get("action") not in (
                    "confirm", "cancel"
                ):
                    raise ValueError("标定回复必须是确认或取消")
                if response["action"] == "cancel":
                    self.cancelled.set()
                    return
                if type(response.get("prompt_id")) is not int:
                    raise ValueError("标定回复缺少步骤编号")
                self.responses.put(response)
            except ValueError as error:
                self.responses.put(error)
        self.cancelled.set()

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise InterruptedError("本次标定已取消")

    def notify(self, step, *, prompt_id=None):
        print(json.dumps(dict(step=step, prompt_id=prompt_id)), file=self.output, flush=True)

    def confirm(self, step):
        self.check_cancelled()
        self.sequence += 1
        self.notify(step, prompt_id=self.sequence)
        while True:
            self.check_cancelled()
            try:
                response = self.responses.get(timeout=0.1)
            except Empty:
                continue
            if isinstance(response, ValueError):
                raise response
            if response["prompt_id"] != self.sequence:
                raise ValueError("标定步骤已变化，请按当前页面操作")
            return


def run_gui_calibration(**options):
    from alohamini.calibration.procedure import calibrate

    interaction = CalibrationInteraction(sys.stdin, sys.stdout)
    with redirect_stdout(sys.stderr):
        try:
            calibrate(**options, interaction=interaction)
        except InterruptedError:
            interaction.notify("cancelled")
            return 0
        except (OSError, ValueError, RuntimeError) as error:
            print(str(error), file=sys.stderr, flush=True)
            interaction.notify("failed")
            return 1
    interaction.notify("succeeded")
    return 0
