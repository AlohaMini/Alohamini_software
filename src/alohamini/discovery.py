"""使用标准 DNS-SD 广播和发现机器人；不打开硬件或发送运动命令。"""

from __future__ import annotations

import json
import logging
import math
import signal
import socket
import sys
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from alohamini.model import get_robot_model
from alohamini.paths import WorkspacePaths

if TYPE_CHECKING:
    from zeroconf import ServiceInfo

SERVICE_TYPE = "_alohamini._tcp.local."
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceIdentity:
    device_id: str
    name: str

    def __post_init__(self):
        if not isinstance(self.device_id, str):
            raise ValueError("设备编号必须是 UUID 字符串")
        object.__setattr__(self, "device_id", str(UUID(self.device_id)))
        if (
            not isinstance(self.name, str)
            or not 1 <= len(self.name.encode("utf-8")) <= 128
            or any(ord(char) < 32 for char in self.name)
        ):
            raise ValueError("设备名称须为 1–128 字节，不能包含控制字符")


@dataclass(frozen=True)
class DiscoveredRobot:
    device_id: str
    name: str
    robot_model: str
    addresses: tuple[str, ...]
    state_port: int
    command_port: int


def load_identity(paths: WorkspacePaths, name: str | None = None) -> DeviceIdentity:
    """编号首次启动时生成并保存，重启和修改名称不改变编号。"""
    path = paths.device_identity
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        identity = DeviceIdentity(str(uuid4()), name if name is not None else socket.gethostname())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"schema_version": 1, "device_id": identity.device_id, "name": identity.name},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return identity
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        raise ValueError("设备身份文件格式无效，不会重新生成或覆盖编号")
    if data["schema_version"] != 1:
        raise ValueError("不支持此设备身份文件版本")
    identity = DeviceIdentity(data.get("device_id"), name if name is not None else data.get("name"))
    if identity.name != data.get("name"):
        data["name"] = identity.name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return identity


@contextmanager
def advertise_robot(identity: DeviceIdentity, robot_model: str, *, bind_host="0.0.0.0"):
    from zeroconf import IPVersion, NonUniqueNameException, ServiceInfo, Zeroconf

    get_robot_model(robot_model)
    if bind_host == "0.0.0.0":
        import ifaddr

        addresses = sorted(
            {
                address.ip
                for adapter in ifaddr.get_adapters()
                for address in adapter.ips
                if isinstance(address.ip, str) and not address.ip.startswith("127.")
            }
        )
    else:
        addresses = [socket.gethostbyname(bind_host)]
    if not addresses:
        raise OSError("没有可广播的 IPv4 网络接口，请检查机器人网络")
    instance = f"AlohaMini-{UUID(identity.device_id).hex}"
    service = ServiceInfo(
        SERVICE_TYPE,
        f"{instance}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(address) for address in addresses],
        port=22,
        properties={
            "txtvers": "2",
            "transport": "ssh",
            "state_port": "5556",
            "device_id": identity.device_id,
            "name": identity.name,
            "robot_model": robot_model,
            "command_port": "5555",
        },
        server=f"{instance.lower()}.local.",
    )
    with ExitStack() as cleanup:
        zeroconf = Zeroconf(interfaces=addresses, ip_version=IPVersion.V4Only)
        cleanup.callback(zeroconf.close)
        # 同一编号的重复广播必须报错，不能悄悄改名后伪装成两台设备。
        try:
            zeroconf.register_service(service, allow_name_change=False)
        except NonUniqueNameException as error:
            raise ValueError(
                "局域网中已有相同编号的机器人，请检查设备身份文件是否被复制"
            ) from error
        cleanup.callback(zeroconf.unregister_service, service)
        logger.info(
            "已广播机器人：%s，编号 %s，管理地址 %s:22",
            identity.name,
            identity.device_id,
            ", ".join(addresses),
        )
        yield identity


def advertise_device(paths: WorkspacePaths, robot_model: str, *, name=None, bind_host="0.0.0.0"):
    """常驻发布设备身份，与电机 Host 的启停无关。"""
    identity = load_identity(paths, name)
    with ExitStack() as cleanup:
        for signum in (signal.SIGTERM, signal.SIGINT):
            # 信号处理器不获取线程锁，由退出异常触发上下文清理。
            previous = signal.signal(signum, lambda _signum, _frame: sys.exit(0))
            cleanup.callback(signal.signal, signum, previous)
        cleanup.enter_context(advertise_robot(identity, robot_model, bind_host=bind_host))
        threading.Event().wait()


def parse_service(info: ServiceInfo) -> DiscoveredRobot:
    """DNS-SD 数据在进入 SDK 时校验一次，之后向调用方返回已解析的值。"""
    from zeroconf import IPVersion

    properties = info.decoded_properties
    if (
        info.type != SERVICE_TYPE
        or properties.get("txtvers") != "2"
        or properties.get("transport") != "ssh"
        or info.port != 22
    ):
        raise ValueError("不支持的机器人广播类型或版本")
    identity = DeviceIdentity(properties["device_id"], properties["name"])
    model = properties["robot_model"]
    get_robot_model(model)
    command_port = int(properties["command_port"])
    state_port = int(properties["state_port"])
    if not 1 <= state_port <= 65535:
        raise ValueError("机器人状态端口无效")
    if not 1 <= command_port <= 65535:
        raise ValueError("机器人控制端口无效")
    addresses = tuple(sorted(info.parsed_addresses(IPVersion.V4Only)))
    if not addresses:
        raise ValueError("机器人广播没有 IPv4 地址")
    return DiscoveredRobot(
        identity.device_id, identity.name, model, addresses, state_port, command_port
    )


def discover_robots(timeout_s=3.0, *, interfaces=None) -> list[DiscoveredRobot]:
    from zeroconf import InterfaceChoice, IPVersion, ServiceBrowser, ServiceListener, Zeroconf

    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 10:
        raise ValueError("发现等待时间须大于 0 且不超过 10 秒")
    records = {}
    stopping = threading.Event()

    class Listener(ServiceListener):
        def add_service(self, zeroconf, service_type, name):
            if stopping.is_set():
                return
            info = zeroconf.get_service_info(service_type, name, timeout=300)
            if info is None:
                return
            try:
                records[name] = parse_service(info)
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                records.pop(name, None)
                logger.warning("忽略无效机器人广播 %s：%s", name, error)

        def update_service(self, zeroconf, service_type, name):
            self.add_service(zeroconf, service_type, name)

        def remove_service(self, zeroconf, service_type, name):
            records.pop(name, None)

    with ExitStack() as cleanup:
        zeroconf = Zeroconf(
            interfaces=InterfaceChoice.All if interfaces is None else interfaces,
            ip_version=IPVersion.V4Only,
        )
        cleanup.callback(zeroconf.close)
        browser = ServiceBrowser(zeroconf, SERVICE_TYPE, Listener())
        cleanup.callback(browser.cancel)
        stopping.wait(timeout_s)
        stopping.set()
    # cancel 已结束监听回调；按编号合并同一设备的多网卡记录。
    devices = {}
    for robot in records.values():
        previous = devices.get(robot.device_id)
        addresses = tuple(sorted(set(robot.addresses + (previous.addresses if previous else ()))))
        devices[robot.device_id] = DiscoveredRobot(
            robot.device_id,
            robot.name,
            robot.robot_model,
            addresses,
            robot.state_port,
            robot.command_port,
        )
    return sorted(devices.values(), key=lambda robot: (robot.name.casefold(), robot.device_id))
