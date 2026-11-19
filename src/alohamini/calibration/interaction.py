"""桌面主臂标定入口，复用共享交互通道。"""

import sys
from contextlib import redirect_stdout

from alohamini.interaction import DesktopInteraction


def run_gui_calibration(**options):
    from alohamini.calibration.procedure import calibrate

    interaction = DesktopInteraction(sys.stdin, sys.stdout)
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
