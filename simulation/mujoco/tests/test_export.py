import io
from types import SimpleNamespace

import mujoco
import numpy as np
from alohamini_mujoco.cli import export_robot
from alohamini_mujoco.meshes import Mesh
from PIL import Image


def test_textured_mesh_export_remains_loadable_after_relocation(tmp_path):
    texture = io.BytesIO()
    Image.new("RGB", (4, 4), (120, 40, 20)).save(texture, format="PNG")
    mesh = Mesh(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )
    xml = """<mujoco><compiler/>
      <asset>
        <mesh name="part" file="part.msh"/>
        <texture name="color" type="2d" file="color.png"/>
        <material name="paint" texture="color"/>
      </asset>
      <worldbody><geom type="mesh" mesh="part" material="paint"/></worldbody>
    </mujoco>"""
    assets = {"part.msh": mesh.msh_bytes(), "color.png": texture.getvalue()}
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    robot = SimpleNamespace(xml=xml, assets=assets, model=model, manifest={})
    exported = export_robot(robot, {}, tmp_path / "before")
    relocated = tmp_path / "after"
    exported.rename(relocated)
    reloaded = mujoco.MjModel.from_xml_path(str(relocated / "model.xml"))
    np.testing.assert_array_equal(reloaded.tex_data, model.tex_data)
    np.testing.assert_array_equal(reloaded.mesh_vert, model.mesh_vert)
    assert reloaded.ntex == reloaded.nmesh == reloaded.ngeom == 1
