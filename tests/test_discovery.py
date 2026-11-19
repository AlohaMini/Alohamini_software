"""真实 DNS-SD 报文与本地身份文件测试，不打开串口或启动电机。"""

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import ifaddr
import pytest
from zeroconf import IPVersion, ServiceInfo, Zeroconf

from alohamini.discovery import (
    SERVICE_TYPE,
    DeviceIdentity,
    advertise_robot,
    discover_robots,
    load_identity,
    parse_service,
)
from alohamini.paths import WorkspacePaths


def test_identity_survives_restart_and_rename(tmp_path):
    paths = WorkspacePaths(tmp_path)
    first = load_identity(paths, "实验室机器人")
    assert load_identity(paths) == first
    renamed = load_identity(paths, "演示机器人")
    assert renamed.device_id == first.device_id
    assert load_identity(paths) == renamed


@pytest.mark.parametrize(
    "content", ["{", "{}", '{"schema_version": 2}', '{"schema_version": 1, "device_id": "损坏"}']
)
def test_invalid_identity_is_never_replaced(tmp_path, content):
    paths = WorkspacePaths(tmp_path)
    paths.device_identity.write_text(content)
    with pytest.raises(ValueError):
        load_identity(paths)
    assert paths.device_identity.read_text() == content


@pytest.mark.parametrize("name", ["", "换\n行", "长" * 43])
def test_invalid_name_does_not_create_identity(tmp_path, name):
    paths = WorkspacePaths(tmp_path)
    with pytest.raises(ValueError):
        load_identity(paths, name)
    assert not paths.device_identity.exists()


def service(**properties):
    return ServiceInfo(
        SERVICE_TYPE,
        f"测试.{SERVICE_TYPE}",
        addresses=[socket.inet_aton("127.0.0.1")],
        port=22,
        properties={
            "txtvers": "2",
            "transport": "ssh",
            "state_port": "5556",
            "device_id": str(uuid4()),
            "name": "机器人",
            "robot_model": "alohamini2pro",
            "command_port": "5555",
            **properties,
        },
    )


def test_service_parses_identity_and_endpoint():
    info = service()
    robot = parse_service(info)
    assert robot.device_id == info.decoded_properties["device_id"]
    assert robot.name == "机器人"
    assert robot.addresses == ("127.0.0.1",)
    assert (robot.state_port, robot.command_port) == (5556, 5555)


@pytest.mark.parametrize(
    "properties",
    [
        {"txtvers": "1"},
        {"transport": "http"},
        {"state_port": "0"},
        {"device_id": "错误编号"},
        {"name": ""},
        {"robot_model": "其他型号"},
        {"command_port": "0"},
        {"command_port": "不是端口"},
    ],
)
def test_malformed_service_is_rejected(properties):
    with pytest.raises((ValueError, KeyError)):
        parse_service(service(**properties))


@pytest.mark.parametrize("timeout", [0, -1, 11, float("nan"), True])
def test_discovery_wait_is_bounded(timeout):
    with pytest.raises(ValueError):
        discover_robots(timeout)


@pytest.fixture
def network_address():
    # Linux 和 Windows 的回环接口不保证支持组播，使用真实网卡验证 DNS-SD。
    if os.environ.get("ALOHAMINI_DISCOVERY_TEST_INTERFACE"):
        return os.environ["ALOHAMINI_DISCOVERY_TEST_INTERFACE"]
    return next(
        address.ip
        for adapter in ifaddr.get_adapters()
        for address in adapter.ips
        if isinstance(address.ip, str) and not address.ip.startswith("127.")
    )


@pytest.mark.multicast
def test_two_same_name_devices_and_service_shutdown(network_address):
    first = DeviceIdentity(str(uuid4()), "同名机器人")
    second = DeviceIdentity(str(uuid4()), "同名机器人")
    with advertise_robot(first, "alohamini2pro", bind_host=network_address):
        with advertise_robot(second, "alohamini2", bind_host=network_address):
            robots = discover_robots(1, interfaces=[network_address])
            assert {first.device_id, second.device_id} <= {robot.device_id for robot in robots}
        robots = discover_robots(1, interfaces=[network_address])
        assert first.device_id in {robot.device_id for robot in robots}
        assert second.device_id not in {robot.device_id for robot in robots}
    assert first.device_id not in {
        robot.device_id for robot in discover_robots(0.5, interfaces=[network_address])
    }


@pytest.mark.multicast
def test_advertiser_failure_unregisters_broadcast(network_address):
    identity = DeviceIdentity(str(uuid4()), "启动失败的机器人")
    with pytest.raises(RuntimeError, match="硬件启动失败"):
        with advertise_robot(identity, "alohamini2pro", bind_host=network_address):
            raise RuntimeError("硬件启动失败")
    assert identity.device_id not in {
        robot.device_id for robot in discover_robots(0.5, interfaces=[network_address])
    }


@pytest.mark.multicast
def test_duplicate_identity_is_an_error_instead_of_silent_rename(network_address):
    identity = DeviceIdentity(str(uuid4()), "重复编号")
    with advertise_robot(identity, "alohamini2pro", bind_host=network_address):
        with pytest.raises(ValueError, match="相同编号"):
            with advertise_robot(identity, "alohamini2pro", bind_host=network_address):
                pytest.fail("不应接受重复编号")


@pytest.mark.multicast
def test_browser_ignores_unsupported_broadcast(network_address):
    zeroconf = Zeroconf(interfaces=[network_address], ip_version=IPVersion.V4Only)
    info = service(txtvers="999")
    try:
        zeroconf.register_service(info)
        assert info.decoded_properties["device_id"] not in {
            robot.device_id for robot in discover_robots(0.5, interfaces=[network_address])
        }
    finally:
        zeroconf.unregister_service(info)
        zeroconf.close()


@pytest.mark.multicast
@pytest.mark.skipif(os.name == "nt", reason="SIGTERM 用于类 Unix 常驻服务；Windows 客户端单独验证")
def test_standalone_advertiser_uses_stable_identity_without_importing_hardware(
    tmp_path, network_address
):
    identity = load_identity(WorkspacePaths(tmp_path), "待启动的机器人")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import importlib.abc,sys
class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('alohamini.hardware', 'alohamini.runtime')):
            raise AssertionError('发现服务不应加载硬件层')
sys.meta_path.insert(0,NoHardware())
from alohamini.cli import main
raise SystemExit(main(sys.argv[1:]))
""",
            "advertise",
            "--robot_model",
            "alohamini2pro",
            "--bind_host",
            network_address,
        ],
        env={
            **os.environ,
            "ALOHAMINI_WORKSPACE": str(tmp_path),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert "已广播机器人" in process.stderr.readline()
        robots = discover_robots(1, interfaces=[network_address])
        assert identity.device_id in {robot.device_id for robot in robots}
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
        assert identity.device_id not in {
            robot.device_id for robot in discover_robots(0.5, interfaces=[network_address])
        }
        assert load_identity(WorkspacePaths(tmp_path)).device_id == identity.device_id
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
