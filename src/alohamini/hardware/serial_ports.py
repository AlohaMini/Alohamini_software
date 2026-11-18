"""三端共用的 USB 串口枚举，保留驱动可直接打开的设备名。"""

import sys
from pathlib import Path


def serial_port_key(device):
    # COM 端口是设备名，转换成文件绝对路径后无法交给串口驱动。
    return device.upper() if sys.platform == "win32" else str(Path(device).resolve())


def usb_ports():
    from serial.tools import list_ports

    return {
        serial_port_key(port.device): port
        for port in list_ports.comports(include_links=False)
        if port.vid is not None and port.pid is not None
    }
