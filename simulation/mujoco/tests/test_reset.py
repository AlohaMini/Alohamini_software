from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.control import FollowerSettings
from alohamini_mujoco.libero import settle_scene


def falling_object(*, x, z):
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option timestep=".001"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <geom name="waiting_robot" type="box" pos="0 0 .1" size=".1 .1 .1"/>
        <body pos="{x} 0 {z}"><freejoint/>
          <geom name="object" type="sphere" size=".03" mass=".1"/>
        </body>
      </worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        model=model,
        data=data,
        robot=SimpleNamespace(geometry={"waiting_robot": {}}),
        settings=FollowerSettings(),
        step=lambda: mujoco.mj_step(model, data),
    )


def test_reset_rejects_a_collision_before_any_physics_step():
    follower = falling_object(x=0, z=0.22)
    with pytest.raises(ValueError, match="Scene reset collision at 0.000 s"):
        settle_scene(follower)
    assert follower.data.time == 0


def test_reset_detects_a_falling_object_hitting_the_waiting_robot():
    follower = falling_object(x=0, z=0.5)
    assert not follower.data.ncon
    with pytest.raises(ValueError, match="waiting_robot.*object"):
        settle_scene(follower)
    assert 0.1 < follower.data.time < 0.5


def test_clear_reset_preserves_the_unmodified_physics():
    follower = falling_object(x=0.5, z=0.5)
    expected = mujoco.MjData(follower.model)
    mujoco.mj_copyData(expected, follower.model, follower.data)
    for _ in range(500):
        mujoco.mj_step(follower.model, expected)
    mujoco.mj_forward(follower.model, expected)
    settle_scene(follower)
    np.testing.assert_array_equal(follower.data.qpos, expected.qpos)
    np.testing.assert_array_equal(follower.data.qvel, expected.qvel)
    assert follower.data.time == pytest.approx(0.5)
    assert follower.data.qpos[2] < 0.031


@pytest.mark.parametrize("duration", [0, -0.1, np.inf, np.nan])
def test_reset_rejects_invalid_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        settle_scene(falling_object(x=0.5, z=0.5), duration)
