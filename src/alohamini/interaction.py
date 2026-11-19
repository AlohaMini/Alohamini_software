"""桌面工作流的逐行 JSON 交互；硬件和数据操作由各应用管理。"""

import json
import threading
from queue import Empty, Queue


class DesktopInteraction:
    def __init__(self, source, output, *, actions=("confirm", "cancel")):
        self.actions = actions
        self.output = output
        self.responses = Queue()
        self.cancelled = threading.Event()
        self.sequence = 0
        threading.Thread(target=self._read, args=(source,), daemon=True).start()

    def _read(self, source):
        for line in source:
            try:
                response = json.loads(line)
                if not isinstance(response, dict) or response.get("action") not in self.actions:
                    raise ValueError("工作流回复包含无效操作")
                if response["action"] in ("cancel", "finish"):
                    self.cancelled.set()
                    return
                if type(response.get("prompt_id")) is not int:
                    raise ValueError("工作流回复缺少步骤编号")
                self.responses.put(response)
            except ValueError as error:
                self.responses.put(error)
        self.cancelled.set()

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise InterruptedError("本次操作已取消")

    def notify(self, step, *, prompt_id=None, **payload):
        print(json.dumps(dict(step=step, prompt_id=prompt_id, **payload)), file=self.output, flush=True)

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
                raise ValueError("操作步骤已变化，请按当前页面操作")
            return
