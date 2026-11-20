"""真实 DNS-SD 报文与本地身份文件测试，不打开串口或启动电机。"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import ifaddr
import pytest
from zeroconf import IPVersion, ServiceInfo, Zeroconf

from alohamini.discovery import (
    SERVICE_TYPE,
    DeviceIdentity,
    _ipv4_addresses,
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


def test_network_addresses_follow_available_interfaces_and_explicit_binding(monkeypatch):
    adapter = ifaddr.Adapter("测试网卡", "测试网卡", [
        ifaddr.IP("192.168.3.18", 24, "测试网卡"),
        ifaddr.IP("192.168.3.19", 24, "测试网卡"),
        ifaddr.IP("127.0.0.1", 8, "回环网卡"),
        ifaddr.IP(("::1", 0, 0), 128, "回环网卡"),
    ])
    monkeypatch.setattr(ifaddr, "get_adapters", lambda: [adapter])
    assert _ipv4_addresses("0.0.0.0") == ["192.168.3.18", "192.168.3.19"]
    assert _ipv4_addresses("192.168.3.18") == ["192.168.3.18"]
    assert _ipv4_addresses("192.168.3.20") == []
    assert _ipv4_addresses("127.0.0.1") == ["127.0.0.1"]
    monkeypatch.setattr(ifaddr, "get_adapters", lambda: [])
    assert _ipv4_addresses("0.0.0.0") == []
    assert _ipv4_addresses("192.168.3.18") == []


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


@pytest.mark.multicast
@pytest.mark.skipif(os.name == "nt", reason="发现守护进程部署在树莓派；Windows 继续验证发现客户端")
@pytest.mark.parametrize("initially_offline", [False, True])
def test_advertiser_refreshes_addresses_and_recovers_without_restart(
    tmp_path, network_address, initially_offline
):
    paths = WorkspacePaths(tmp_path)
    identity = load_identity(paths, "换网测试机器人")
    original_identity = paths.device_identity.read_bytes()
    addresses_file = tmp_path / "network.json"

    def change_addresses(addresses):
        pending = addresses_file.with_suffix(".pending")
        pending.write_text(json.dumps(addresses))
        pending.replace(addresses_file)

    def wait_for_addresses(expected):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert process.poll() is None
            robots = discover_robots(0.3, interfaces=[network_address])
            found = next(
                (robot.addresses for robot in robots if robot.device_id == identity.device_id), None
            )
            if found == expected:
                return
        pytest.fail(f"广播未更新为预期地址：{expected}，实际：{found}")

    change_addresses([] if initially_offline else [network_address])
    process = subprocess.Popen([
        sys.executable, "-c", """
import importlib.abc,json,logging,sys
from pathlib import Path
class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('alohamini.hardware','alohamini.runtime')):
            raise AssertionError('地址刷新不应加载硬件层')
sys.meta_path.insert(0,NoHardware())
from alohamini import discovery
from alohamini.paths import WorkspacePaths
logging.basicConfig(level=logging.INFO, format='%(message)s')
# 只替换操作系统地址来源；注册、网卡绑定和报文收发仍使用真实 Zeroconf。
discovery._ipv4_addresses=lambda _host: sorted(json.loads(Path(sys.argv[1]).read_text()))
discovery.NETWORK_POLL_INTERVAL_S=0.1
discovery.advertise_device(WorkspacePaths(sys.argv[2]),'alohamini2pro',bind_host=sys.argv[3])
""", str(addresses_file), str(tmp_path), network_address],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        stderr=subprocess.PIPE, text=True)
    try:
        if initially_offline:
            assert "等待网络" in process.stderr.readline()
            wait_for_addresses(None)
            change_addresses([network_address])
        wait_for_addresses((network_address,))
        change_addresses([network_address, "127.0.0.1"])
        wait_for_addresses(tuple(sorted([network_address, "127.0.0.1"])))
        change_addresses([network_address])
        wait_for_addresses((network_address,))
        change_addresses([])
        wait_for_addresses(None)
        change_addresses([network_address])
        wait_for_addresses((network_address,))
        assert paths.device_identity.read_bytes() == original_identity
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
        assert identity.device_id not in {
            robot.device_id for robot in discover_robots(0.5, interfaces=[network_address])
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
