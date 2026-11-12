from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.cameras import (
    OPTICAL_TO_OPENGL,
    CameraCapture,
    Lens,
    add_cameras,
    model_matrix,
    resize_matrix,
)
from alohamini_mujoco.collection import CollectionSettings
from alohamini_mujoco.description import Description

from alohamini.model import get_robot_model


def definition(kind="plumb_bob", D=None):
    return {
        "name": "calibrated",
        "body": "mount",
        "T_body_camera_optical": OPTICAL_TO_OPENGL.tolist(),
        "intrinsics": {
            "resolution": [320, 240],
            "K": [[210, 0, 147.2], [0, 195, 128.4], [0, 0, 1]],
            "distortion_model": kind,
            "D": D if D is not None else [0, 0, 0, 0, 0],
            "status": "candidate_intrinsics_requires_review",
        },
    }


def robot_for_camera(camera):
    # Coloured isolated planes make projection errors observable in rendered pixels.
    root = ET.fromstring("""<mujoco>
      <visual><global offwidth="640" offheight="480"/>
      <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <worldbody><body name="mount"/>
      <geom type="box" pos=".3 -.2 -2" size=".04 .04 .001" rgba="1 0 0 1"/>
      <geom type="box" pos="-.5 .3 -2" size=".04 .04 .001" rgba="0 1 0 1"/>
      </worldbody></mujoco>""")
    add_cameras(root, [camera])
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return SimpleNamespace(model=model, scene=SimpleNamespace(configuration={"cameras": [camera]}))


@pytest.mark.parametrize(
    "kind,D",
    [
        ("plumb_bob", [-0.18, 0.035, 0.001, -0.002, 0]),
        ("equidistant", [-0.04, 0.005, 0, 0]),
    ],
)
def test_rendered_raw_pixels_match_independent_opencv_projection(kind, D):
    import cv2

    camera = definition(kind, D)
    robot = robot_for_camera(camera)
    data = mujoco.MjData(robot.model)
    mujoco.mj_forward(robot.model, data)
    K = np.array(camera["intrinsics"]["K"], dtype=float)
    xyz = np.array([[[0.3, 0.2, 2]], [[-0.5, -0.3, 2]]])
    project = cv2.fisheye.projectPoints if kind == "equidistant" else cv2.projectPoints
    expected = project(xyz, np.zeros(3), np.zeros(3), K, np.array(D))[0].reshape(-1, 2)
    with CameraCapture(robot, ["calibrated"], width=320, height=240) as capture:
        image = capture.image(data, "calibrated", mujoco.MjvOption())
        metadata = capture.metadata("calibrated")
        for i, point in enumerate(expected):
            pixels = np.argwhere((image[..., i] > 200) & (image[..., 1 - i] < 30))
            np.testing.assert_allclose(pixels[:, ::-1].mean(0), point, atol=0.8)
        assert metadata["calibration"]["intrinsics"]["status"].startswith("candidate_")
        np.testing.assert_allclose(metadata["K_opencv"], K)
        assert len(capture.renderers) == 1
    assert not capture.renderers


def test_projection_metadata_matches_noncentral_renderer_and_resizing():
    camera = definition()
    robot = robot_for_camera(camera)
    data = mujoco.MjData(robot.model)
    mujoco.mj_forward(robot.model, data)
    with CameraCapture(robot, ["calibrated"], width=160, height=160) as capture:
        image = capture.image(data, "calibrated", mujoco.MjvOption())
        K = np.array(capture.metadata("calibrated")["K_opencv"])
        expected = K @ [0.15, 0.1, 1]
        pixels = np.argwhere((image[..., 0] > 100) & (image[..., 1] < 30))
        np.testing.assert_allclose(pixels[:, ::-1].mean(0), expected[:2], atol=0.8)
    np.testing.assert_allclose(
        resize_matrix([[210, 0, 147.2], [0, 195, 128.4], [0, 0, 1]], (320, 240), (160, 160)), K
    )


def test_rejects_lens_that_cannot_invert_corners():
    lens = Lens(
        {
            "resolution": [640, 480],
            "K": [[389, 0, 330], [0, 388, 257], [0, 0, 1]],
            "distortion_model": "plumb_bob",
            "D": [-0.35375734, 0.15687407, -0.00271585, 0.00070447, -0.03666877],
        }
    )
    with pytest.raises(ValueError, match="cannot invert"):
        lens.mjcf_attributes()


def test_exported_camera_snapshot_is_authoritative_without_live_scene_files():
    robot = robot_for_camera(definition())
    robot.scene = None
    with CameraCapture(robot, ["calibrated"], width=320, height=240) as capture:
        np.testing.assert_allclose(
            capture.metadata("calibrated")["K_opencv"], definition()["intrinsics"]["K"]
        )
        assert "calibrated" in capture.lenses
    robot.scene = SimpleNamespace(configuration={"cameras": [definition()]})
    robot.scene.configuration["cameras"][0]["intrinsics"]["D"][0] = 0.1
    with pytest.raises(ValueError, match="differs from the recorded model"):
        CameraCapture(robot, ["calibrated"], width=320, height=240)


def test_legacy_virtual_camera_matches_direct_renderer():
    camera = {
        "name": "calibrated",
        "body": "mount",
        "fovy_deg": 60,
        "T_body_camera": np.eye(4).tolist(),
    }
    robot = robot_for_camera(camera)
    data = mujoco.MjData(robot.model)
    mujoco.mj_forward(robot.model, data)
    with CameraCapture(robot, ["calibrated"], width=128, height=128) as capture:
        image = capture.image(data, "calibrated", mujoco.MjvOption()).copy()
    with mujoco.Renderer(robot.model, width=128, height=128) as renderer:
        renderer.update_scene(data, camera="calibrated")
        np.testing.assert_array_equal(image, renderer.render())
    K = model_matrix(robot.model, "calibrated", (128, 128))
    np.testing.assert_allclose(K[:2, 2], [63.5, 63.5])


def test_camera_mounts_follow_the_correct_urdf_chain():
    description = Description(get_robot_model("alohamini2pro"))
    low = description.forward({"vertical_move": 0})
    high = description.forward({"vertical_move": 0.2, "root_x_axis_joint": 1})
    np.testing.assert_allclose(
        high["front_camera"][:3, 3] - low["front_camera"][:3, 3], [1, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        high["chest_camera"][:3, 3] - low["chest_camera"][:3, 3], [1, 0, 0.2], atol=1e-12
    )


@pytest.mark.parametrize(
    "value,expected", [(128, (128, 128)), ((480, 640), (480, 640)), ([480, 640], (480, 640))]
)
def test_collection_accepts_native_camera_resolution(value, expected):
    assert CollectionSettings(image_size=value).image_shape == expected


@pytest.mark.parametrize("value", [True, 0, (128,), (20, 640), (480, 1.5), (1, 2, 3)])
def test_collection_rejects_invalid_camera_resolution(value):
    with pytest.raises(ValueError):
        CollectionSettings(image_size=value)


def test_workspace_geometry_gate_detects_clipping_and_behind_camera():
    camera = definition()
    root = ET.fromstring("""<mujoco><worldbody><body name="mount"/>
      <body name="object" pos="0 0 -2"><geom name="object_mesh" type="box"
      size=".1 .1 .1" group="2"/></body></worldbody></mujoco>""")
    add_cameras(root, [camera])
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    robot = SimpleNamespace(model=model, scene=None)
    data = mujoco.MjData(model)
    with CameraCapture(robot, ["calibrated"], width=320, height=240) as capture:
        for position, expected in [([0, 0, -2], True), ([2, 0, -2], False), ([0, 0, 2], False)]:
            model.body("object").pos[:] = position
            mujoco.mj_forward(model, data)
            bounds, passed = capture.object_bounds(data, "calibrated", "object")
            assert passed is expected
            assert bounds.shape == (4,)
