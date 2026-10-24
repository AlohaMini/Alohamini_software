import mujoco
import numpy as np
import pytest
from alohamini_mujoco.native_distance import InvalidNativeDistance, checked_geom_distance


def fixture_model(shape):
    mesh = """<asset><mesh name="box" vertex="
      -.02 -.03 -.04  .02 -.03 -.04  -.02 .03 -.04  .02 .03 -.04
      -.02 -.03  .04  .02 -.03  .04  -.02 .03  .04  .02 .03  .04"/></asset>"""
    geometry = {
        "box": 'type="box" size=".02 .03 .04"',
        "sphere": 'type="sphere" size=".02"',
        "cylinder": 'type="cylinder" size=".02 .04"',
        "capsule": 'type="capsule" size=".02 .04"',
        "ellipsoid": 'type="ellipsoid" size=".02 .03 .04"',
        "mesh": 'type="mesh" mesh="box"',
    }[shape]
    xml = f"""<mujoco>{mesh}<worldbody>
      <body pos=".1 -.2 .3" euler="20 -30 40"><geom name="shape" {geometry}/></body>
      <body name="probe" mocap="true"><geom name="probe" size=".002"/></body>
      </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("shape", ["box", "sphere", "cylinder", "capsule", "ellipsoid", "mesh"])
def test_actual_native_distances_agree_with_analytic_axis_clearance(shape):
    model, data = fixture_model(shape)
    geom, probe = model.geom("shape").id, model.geom("probe").id
    for clearance in [0.001, -0.001]:
        center = np.array([0.02 + 0.002 + clearance, 0, 0])
        data.mocap_pos[0] = data.geom_xmat[geom].reshape(3, 3) @ center + data.geom_xpos[geom]
        mujoco.mj_forward(model, data)
        distance, witness = checked_geom_distance(model, data, geom, probe, 0.01)
        assert distance == pytest.approx(clearance, abs=1e-6)
        assert np.linalg.norm(witness[3:] - witness[:3]) == pytest.approx(abs(distance), abs=1e-8)
        reverse, _ = checked_geom_distance(model, data, probe, geom, 0.01)
        assert reverse == pytest.approx(distance, abs=1e-7)


@pytest.mark.parametrize("damage", ["witness", "false_zero", "excess_penetration", "nan", "cutoff"])
def test_rejects_contradictions_instead_of_repairing_the_engine_result(monkeypatch, damage):
    model, data = fixture_model("box")
    geom, probe = model.geom("shape").id, model.geom("probe").id
    R, center = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
    clearance = -0.001 if damage == "excess_penetration" else 0.005
    data.mocap_pos[0] = R @ [0.022 + clearance, 0, 0] + center
    mujoco.mj_forward(model, data)

    def broken(model, data, first, second, maximum, witness):
        witness[:] = 0
        if damage == "witness":
            witness[3] = 0.005
            return 0.0
        if damage == "excess_penetration":
            witness[3] = 0.1
            return -0.1
        if damage == "nan":
            return float("nan")
        if damage == "cutoff":
            return maximum * 2
        return 0.0  # Consistent zero witnesses still contradict a separating plane.

    monkeypatch.setattr(mujoco, "mj_geomDistance", broken)
    with pytest.raises(InvalidNativeDistance, match="Invalid MuJoCo"):
        checked_geom_distance(model, data, geom, probe, 0.01)


def test_censored_distance_has_no_required_witnesses():
    model, data = fixture_model("sphere")
    data.mocap_pos[0] = [10, 0, 0]
    mujoco.mj_forward(model, data)
    distance, witness = checked_geom_distance(
        model, data, model.geom("shape").id, model.geom("probe").id, 0.01
    )
    assert distance == 0.01
    np.testing.assert_array_equal(witness, np.zeros(6))


@pytest.mark.parametrize("cutoff", [0, -1, True, float("nan"), float("inf")])
def test_requires_positive_finite_cutoff(cutoff):
    model, data = fixture_model("sphere")
    with pytest.raises(ValueError, match="cutoff"):
        checked_geom_distance(model, data, 0, 1, cutoff)
