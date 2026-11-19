"""真实 DNS-SD 报文与本地身份文件测试，不打开串口或启动电机。"""

import socket
from uuid import uuid4

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
        port=5556,
        properties={
            "txtvers": "1",
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
        {"txtvers": "2"},
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


def test_two_same_name_devices_and_service_shutdown():
    first = DeviceIdentity(str(uuid4()), "同名机器人")
    second = DeviceIdentity(str(uuid4()), "同名机器人")
    with advertise_robot(first, "alohamini2pro", bind_host="127.0.0.1"):
        with advertise_robot(second, "alohamini2", bind_host="127.0.0.1"):
            robots = discover_robots(1, interfaces=["127.0.0.1"])
            assert {robot.device_id for robot in robots} == {first.device_id, second.device_id}
        robots = discover_robots(1, interfaces=["127.0.0.1"])
        assert [robot.device_id for robot in robots] == [first.device_id]
    assert discover_robots(0.5, interfaces=["127.0.0.1"]) == []


def test_host_failure_unregisters_broadcast():
    identity = DeviceIdentity(str(uuid4()), "启动失败的机器人")
    with pytest.raises(RuntimeError, match="硬件启动失败"):
        with advertise_robot(identity, "alohamini2pro", bind_host="127.0.0.1"):
            raise RuntimeError("硬件启动失败")
    assert discover_robots(0.5, interfaces=["127.0.0.1"]) == []


def test_duplicate_identity_is_an_error_instead_of_silent_rename():
    identity = DeviceIdentity(str(uuid4()), "重复编号")
    with advertise_robot(identity, "alohamini2pro", bind_host="127.0.0.1"):
        with pytest.raises(ValueError, match="相同编号"):
            with advertise_robot(identity, "alohamini2pro", bind_host="127.0.0.1"):
                pytest.fail("不应接受重复编号")


def test_browser_ignores_unsupported_broadcast():
    zeroconf = Zeroconf(interfaces=["127.0.0.1"], ip_version=IPVersion.V4Only)
    info = service(txtvers="999")
    try:
        zeroconf.register_service(info)
        assert discover_robots(0.5, interfaces=["127.0.0.1"]) == []
    finally:
        zeroconf.unregister_service(info)
        zeroconf.close()
