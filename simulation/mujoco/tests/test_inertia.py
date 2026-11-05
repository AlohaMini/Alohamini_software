import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.frames import transform
from alohamini_mujoco.inertia import (
    MassProperties,
    combine,
    parse_solidworks_reports,
    subtract,
)


def report_text():
    rotation = transform(rpy=[0.3, -0.4, 0.5])[:3, :3]
    central = rotation @ np.diag([0.002, 0.003, 0.004]) @ rotation.T
    com = np.array([0.1, 0.2, 0.3])
    origin = central + 1.5 * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    lines = [
        "部件 的质量属性",
        "配置： 默认",
        "坐标系： -- 默认 --",
        "质量 = 1500.00 克",
        "体积 = 123456.78 立方毫米",
        "表面积 = 12345.67 平方毫米",
        "质心 : ( 毫米 )",
        "X = 100.00",
        "Y = 200.00",
        "Z = 300.00",
        "惯性主轴和主惯性矩: ( 克 * 平方毫米 )",
        "Px = 2000000.00",
        "Py = 3000000.00",
        "Pz = 4000000.00",
    ]
    for prefix, matrix in [("L", central), ("I", origin)]:
        lines.append("惯性张量：（使用正张量记数法。）")
        for i, a in enumerate("xyz"):
            for j, b in enumerate("xyz"):
                value = matrix[i, j] * 1e9 * (1 if i == j else -1)
                lines.append(f"{prefix}{a}{b} = {value:.2f}")
    return "\n".join(lines) + "\n", central


def test_report_units_sign_and_transformed_inertias_survive_native_compilation():
    text, central = report_text()
    record = parse_solidworks_reports(text)[0]
    T = transform([0.4, -0.2, 0.1], [0.2, 0.7, -0.3])
    bound = record.properties.in_frame(T, "bound_body")
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", inertiafromgeom="false", balanceinertia="false")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=bound.frame)
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "inertial", **bound.mjcf_attributes())
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = model.body(bound.frame).id
    R = data.ximat[body_id].reshape(3, 3)
    np.testing.assert_allclose(model.body_mass[body_id], 1.5)
    np.testing.assert_allclose(data.xipos[body_id], T[:3, :3] @ [0.1, 0.2, 0.3] + T[:3, 3])
    np.testing.assert_allclose(
        R @ np.diag(model.body_inertia[body_id]) @ R.T,
        T[:3, :3] @ central @ T[:3, :3].T,
        atol=1e-11,
        rtol=0,
    )
    for _ in range(20):
        mujoco.mj_step(model, data)
    assert data.qvel[2] == pytest.approx(model.opt.gravity[2] * model.opt.timestep * 20)
    np.testing.assert_allclose(data.qvel[3:], 0, atol=1e-13)


def test_composition_and_subtraction_recover_two_displaced_spheres():
    a = MassProperties("assembly", 1, [-0.5, 0, 0], np.eye(3) * 0.004)
    b = MassProperties("assembly", 1, [0.5, 0, 0], np.eye(3) * 0.004)
    total = combine([a, b])
    assert total.mass_kg == 2
    np.testing.assert_array_equal(total.com_m, 0)
    np.testing.assert_allclose(total.central_inertia_kg_m2, np.diag([0.008, 0.508, 0.508]))
    recovered = subtract(total, [b])
    assert recovered.mass_kg == 1
    np.testing.assert_allclose(recovered.com_m, a.com_m)
    np.testing.assert_allclose(recovered.central_inertia_kg_m2, a.central_inertia_kg_m2)
    with pytest.raises(ValueError, match="Nonphysical"):
        subtract(a, [MassProperties("assembly", 0.9, [0.5, 0, 0], np.eye(3) * 0.001)])


def test_duplicate_default_frame_names_cannot_be_combined_without_a_binding():
    text, _ = report_text()
    first, second = parse_solidworks_reports(text + text)
    assert first.name == second.name
    assert first.coordinate_system == second.coordinate_system == "-- 默认 --"
    assert first.record_id != second.record_id
    with pytest.raises(ValueError, match="explicit frame"):
        combine([first.properties, second.properties])
    total = combine(
        [
            first.properties.in_frame(np.eye(4), "link"),
            second.properties.in_frame(np.eye(4), "link"),
        ]
    )
    assert total.mass_kg == 3
    other_document = parse_solidworks_reports(text.replace("部件 的", "另一部件 的"))[0]
    assert other_document.properties.frame != parse_solidworks_reports(text)[0].properties.frame


def test_report_rejects_a_wrong_origin_tensor_instead_of_importing_it_as_central():
    text, _ = report_text()
    text = text.replace("Ixx =", "Ixx = 1.00\nignored =")
    with pytest.raises(ValueError, match="parallel-axis"):
        parse_solidworks_reports(text)


def test_solidworks_overridden_mass_with_real_printed_rounding():
    text = """B_F_J2_Rotation 的质量属性
     配置： 默认
     坐标系： -- 默认 --
质量（用户覆盖） = 73.33 克
体积 = 133377.85 立方毫米
表面积 = 32844.06 平方毫米
质心 : ( 毫米 )
    X = -16.19
    Y = -48.84
    Z = 67.05
惯性主轴和主惯性矩: ( 克 * 平方毫米 )
    Ix = (-0.03, 0.99, -0.12) Px = 37024.37
    Iy = (-1.00, -0.04, -0.02) Py = 58872.31
    Iz = (-0.02, 0.11, 0.99) Pz = 71094.37
由重心决定，并且对齐输出的坐标系。 （使用正张量记数法。）
    Lxx = 58855.04 Lxy = -689.54 Lxz = 374.53
    Lyx = -689.54 Lyy = 37499.61 Lyz = -3891.84
    Lzx = 374.53 Lzy = -3891.84 Lzz = 70636.41
由输出座标系决定。 （使用正张量记数法。）
    Ixx = 563426.12 Ixy = 57304.44 Ixz = -79241.97
    Iyx = 57304.44 Iyy = 386385.69 Iyz = -244020.02
    Izx = -79241.97 Izy = -244020.02 Izz = 264778.18
"""
    record = parse_solidworks_reports(text)[0]
    assert record.mass_overridden
    assert record.properties.mass_kg == pytest.approx(0.07333)
    np.testing.assert_allclose(record.properties.com_m, [-0.01619, -0.04884, 0.06705])
    assert record.properties.central_inertia_kg_m2[0, 1] == pytest.approx(689.54e-9)
    assert record.to_dict()["mass_overridden"] is True
    assert not parse_solidworks_reports(report_text()[0])[0].mass_overridden
