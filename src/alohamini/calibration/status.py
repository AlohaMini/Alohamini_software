"""查询当前主臂和标定记录。查询不会打开电机、登记设备或迁移历史文件。"""

from importlib import import_module

from alohamini.environment import runtime_environment
from alohamini.hardware import usb_setup
from alohamini.hardware.serial_ports import serial_port_key
from alohamini.model import get_robot_model

from .reference import calibration_motors, load_reference_profile
from .servo import load_motor_calibration


def check_leader_environment():
    with runtime_environment(["calibrate", "leader"]):
        import_module("scservo_sdk.protocol_packet_handler")
        return usb_setup.usb_ports()


def inspect_leader(workspace, robot_model, ports=None, leader_id="am_leader_bi"):
    available = {}
    connected = False

    def result(state, label, message, detail=""):
        return {"state": state, "label": label, "message": message, "detail": detail,
                "success": state == "ready", "connected": connected,
                "available_ports": sorted(available)}

    try:
        available = check_leader_environment()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return result("unavailable", "需要准备环境", "请先准备主臂标定环境。", str(error))
    if ports is not None and (not all(ports.values())
                              or serial_port_key(ports["left"]) == serial_port_key(ports["right"])):
        return result("identify", "待识别", "请为左右主臂选择两个不同的接口。")
    try:
        directory = usb_setup.registered_leader_directory(workspace, ports, ("left", "right"), available)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return result("identify", "待识别", "请检查左右主臂的接线和接口，首次使用需登记。", str(error))
    if directory is None:
        return result("connect", "等待连接", "请接上左右主臂。首次使用需登记这对主臂。")
    connected = True
    names = [workspace.calibration_file("teleoperators", f"{leader_id}_{side}").name
             for side in ("left", "right")]
    try:
        files = [directory / name for name in names]
        if not any(path.exists() for path in files):
            bindings = usb_setup.legacy_leader_bindings(workspace, leader_id)
            if bindings is None or usb_setup._pair_key(bindings) != directory.name:
                return result("new", "待标定", "已识别这对主臂，摆好 Home 后即可标定。")
            files = [workspace.calibration / "teleoperators" / name for name in names]
        if not all(path.is_file() for path in files):
            return result("invalid", "需要重新标定", "这对主臂的标定记录不完整，请重新标定。")
        left, right = (load_reference_profile(path, robot_model) for path in files)
        if left is None and right is None:
            model = get_robot_model(robot_model)
            for side, path in zip(("left", "right"), files, strict=True):
                load_motor_calibration(path, calibration_motors(model, side, leader=True))
            return result("legacy", "需要重新标定", "升级后需要重新标定一次，请重新标定这对主臂。")
        if left != right or left["role"] != "leader":
            raise ValueError("Home calibration records do not match the current leader pair")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return result("invalid", "需要重新标定", "Home 记录校验失败，请重新标定这对主臂。", str(error))
    return result("ready", "已标定", "这对主臂已完成 Home 标定，可以使用。")
