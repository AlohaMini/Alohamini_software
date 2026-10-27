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


def test_separated_thin_boxes_do_not_become_a_false_gjk_contact():
    """Geometry from upstream 35342f4: an early-exit regression in GJK."""
    # https://github.com/google-deepmind/mujoco/commit/35342f4
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="box" size=".015108062742470629 .00020768330936692047 .00033708479398415232"/>
      <geom type="box" size=".0022889234034685867 .00039019141130537644 .00066943745945985227"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Set native query poses directly to preserve the numerical reproduction.
    rotations = np.array(
        [
            [
                [0.41747699392674587, 0.7445704580736479, -0.5208913442416622],
                [0.735898261710082, -0.6133129289446175, -0.28688150794614425],
                [-0.5330728917830726, -0.263556605216739, -0.8039721437292835],
            ],
            [
                [-0.11330328891322261, 0.6014637637569688, -0.7908246996703808],
                [0.9557435215667675, 0.2834865022971088, 0.07867479900517776],
                [0.2715081687621951, -0.7469114699230777, -0.6069650487406557],
            ],
        ]
    )
    data.geom_xmat[:] = rotations.reshape(2, 9)
    data.geom_xpos[1] = [-0.004522410871927232, -0.009955852699440446, 0.00749543399075938]
    axes = np.vstack([rotations[0].T, rotations[1].T])
    axes = np.vstack([axes, np.cross(rotations[0].T[:, None], rotations[1].T).reshape(-1, 3)])
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-15] / lengths[lengths > 1e-15, None]
    centers = data.geom_xpos @ axes.T
    radii = np.array(
        [abs(axes @ R) @ size for R, size in zip(rotations, model.geom_size, strict=True)]
    )
    gap = float((abs(centers[1] - centers[0]) - radii.sum(0)).max())
    assert gap > 0  # Independent SAT projection proves these boxes are separated.
    distance, _ = checked_geom_distance(model, data, 0, 1, 1.0)
    assert gap - 1e-10 <= distance <= model.opt.ccd_tolerance


@pytest.mark.parametrize("cutoff", [0, -1, True, float("nan"), float("inf")])
def test_requires_positive_finite_cutoff(cutoff):
    model, data = fixture_model("sphere")
    with pytest.raises(ValueError, match="cutoff"):
        checked_geom_distance(model, data, 0, 1, cutoff)
