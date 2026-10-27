import mujoco
import numpy as np
import pytest
from alohamini_mujoco.compilation import COMPILER_CACHE_BYTES, compile_xml, compiler_cache, load_xml
from alohamini_mujoco.meshes import Mesh


def test_compiler_cache_budget_restores_caller_capacity_on_success_and_error():
    cache = mujoco.mj_getCache()
    previous = mujoco.mj_getCacheCapacity(cache)
    with compiler_cache():
        assert mujoco.mj_getCacheCapacity(cache) == min(previous, COMPILER_CACHE_BYTES)
    assert mujoco.mj_getCacheCapacity(cache) == previous
    with pytest.raises(ValueError):
        compile_xml("<mujoco><invalid/></mujoco>")
    assert mujoco.mj_getCacheCapacity(cache) == previous
    try:
        mujoco.mj_setCacheCapacity(cache, 0)
        with compiler_cache():
            assert mujoco.mj_getCacheCapacity(cache) == 0
    finally:
        mujoco.mj_setCacheCapacity(cache, previous)


def test_budgeted_compilation_keeps_native_model_fields_and_relative_asset_loading(tmp_path):
    mesh = Mesh(
        np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.03, 0], [0, 0, 0.04]]),
        np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
    )
    assets = {"part.msh": mesh.msh_bytes()}
    (tmp_path / "part.msh").write_bytes(assets["part.msh"])
    xml = (
        '<mujoco><asset><mesh name="part" file="part.msh"/></asset>'
        '<worldbody><body><freejoint/><inertial mass="1" pos="0 0 0" '
        'diaginertia=".1 .2 .3"/><geom type="mesh" mesh="part" mass="0"/></body>'
        "</worldbody></mujoco>"
    )
    native = mujoco.MjModel.from_xml_string(xml, assets=assets)
    budgeted = compile_xml(xml, assets=assets)
    path = tmp_path / "model.xml"
    path.write_text(xml)
    loaded = load_xml(path)
    for candidate in (budgeted, loaded):
        for field in ("body_mass", "body_inertia", "geom_pos", "mesh_vert", "mesh_face"):
            np.testing.assert_array_equal(getattr(candidate, field), getattr(native, field))
        data = mujoco.MjData(candidate)
        mujoco.mj_step(candidate, data)
        assert np.isfinite(data.qpos).all()
