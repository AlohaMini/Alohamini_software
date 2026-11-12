"""Optical-frame cameras with calibrated intrinsics and raw lens distortion."""

from __future__ import annotations

import json
from functools import cached_property

import mujoco
import numpy as np

from .frames import rigid_transform

OPTICAL_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def resize_matrix(matrix, source_size, target_size):
    """Resize OpenCV integer-centre intrinsics with align_corners=False."""
    result = np.array(matrix, dtype=float, copy=True)
    sx, sy = np.asarray(target_size) / np.asarray(source_size)
    result[0, 0] *= sx
    result[1, 1] *= sy
    result[0, 2] = (result[0, 2] + 0.5) * sx - 0.5
    result[1, 2] = (result[1, 2] + 0.5) * sy - 0.5
    return result


def model_matrix(model, name, size):
    """The actual renderer's OpenCV K, including non-square focal/principal."""
    camera = model.camera(name)
    width, height = size
    sx, sy = model.cam_sensorsize[camera.id]
    if sx > 0 and sy > 0:
        fx, fy, px, py = model.cam_intrinsic[camera.id]
        return np.array(
            [
                [fx / sx * width, 0, (width - 1) / 2 - px / sx * width],
                [0, fy / sy * height, (height - 1) / 2 - py / sy * height],
                [0, 0, 1],
            ],
            dtype=float,
        )
    focal = height / (2 * np.tan(np.deg2rad(float(camera.fovy[0])) / 2))
    return np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])


class Lens:
    """A frozen numeric calibration snapshot; no external file lookup at runtime."""

    def __init__(self, specification):
        self.size = tuple(specification["resolution"])
        if len(self.size) != 2 or any(type(x) is not int or x < 4 for x in self.size):
            raise ValueError("Camera resolution must be [width, height], each >= 4")
        self.K = np.array(specification["K"], dtype=float, copy=True)
        self.D = np.array(specification["D"], dtype=float, copy=True)
        self.kind = specification["distortion_model"]
        if (
            self.K.shape != (3, 3)
            or not np.isfinite(self.K).all()
            or not np.isfinite(self.D).all()
            or self.K[0, 0] <= 0
            or self.K[1, 1] <= 0
            or not np.array_equal(self.K[2], [0, 0, 1])
            or self.K[0, 1] != 0
            or self.K[1, 0] != 0
        ):
            raise ValueError("Camera K must have finite positive focal lengths and zero skew")
        count = {"plumb_bob": 5, "equidistant": 4}.get(self.kind)
        if count is None or self.D.shape != (count,):
            raise ValueError("Use plumb_bob with 5 or equidistant with 4 distortion coefficients")
        self.K.flags.writeable = self.D.flags.writeable = False

    def rays(self, pixels):
        import cv2

        uv = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
        if self.kind == "equidistant":
            xy = cv2.fisheye.undistortPoints(uv, self.K, self.D).reshape(-1, 2)
        else:
            xy = cv2.undistortPointsIter(
                uv,
                self.K,
                self.D,
                None,
                None,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10),
            ).reshape(-1, 2)
        xyz = np.c_[xy, np.ones(len(xy))].reshape(-1, 1, 3)
        project = cv2.fisheye.projectPoints if self.kind == "equidistant" else cv2.projectPoints
        restored = project(xyz, np.zeros(3), np.zeros(3), self.K, self.D)[0]
        if (
            not np.isfinite(xy).all()
            or np.max(np.abs(xy)) > 100
            or np.max(np.linalg.norm(restored - uv, axis=2)) > 0.1
        ):
            raise ValueError(
                "Camera distortion cannot invert the complete image; review calibration"
            )
        return xy

    @cached_property
    def rendering(self):
        """Ideal frustum and inverse warp covering every raw pixel, including corners."""
        width, height = self.size
        y, x = np.indices((height, width))
        xy = self.rays(np.stack([x, y], axis=-1))
        low, high = xy.min(axis=0), xy.max(axis=0)
        focal = (np.array(self.size) - 3) / (high - low)
        center = 1 - low * focal
        K = np.array([[focal[0], 0, center[0]], [0, focal[1], center[1]], [0, 0, 1]])
        mapping = (xy * focal + center).reshape(height, width, 2).astype(np.float32)
        return K, mapping

    def mjcf_attributes(self):
        from .model import text

        width, height = self.size
        K, _ = self.rendering
        # MuJoCo's positive principal shifts the image left/up. Its projection
        # sensor ignores principal offsets in 3.12; tests use the real renderer.
        return {
            "sensorsize": text([width / height, 1]),
            "resolution": f"{width} {height}",
            "focal": text([K[0, 0] / height, K[1, 1] / height]),
            "principal": text(
                [((width - 1) / 2 - K[0, 2]) / height, ((height - 1) / 2 - K[1, 2]) / height]
            ),
        }

    def raw_image(self, ideal_rgb):
        import cv2

        mapping = self.rendering[1]
        return cv2.remap(ideal_rgb, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)


def add_cameras(root, definitions):
    """Add calibrated optical cameras or the existing fovy/OpenGL camera definitions."""
    import xml.etree.ElementTree as ET

    from .model import pose_attributes

    for camera in definitions:
        if camera.get("role") not in (None, "workspace", "wrist", "overview"):
            raise ValueError("Camera role must be workspace, wrist or overview")
        parent = root.find(f".//body[@name='{camera['body']}']")
        if parent is None:
            raise ValueError(f"Unknown camera parent: {camera['body']}")
        if "intrinsics" in camera:
            if "fovy_deg" in camera or "T_body_camera" in camera:
                raise ValueError("Calibrated cameras use T_body_camera_optical and intrinsics")
            lens = Lens(camera["intrinsics"])
            attributes = lens.mjcf_attributes()
            pose = rigid_transform(camera["T_body_camera_optical"]) @ OPTICAL_TO_OPENGL
            custom = root.find("custom")
            if custom is None:
                custom = ET.SubElement(root, "custom")
            # Freeze the complete pixel model in the exported MJCF as well as
            # the episode manifest. Checkpoint model hashes then cover distortion.
            ET.SubElement(
                custom,
                "text",
                name=f"alohamini.camera.{camera['name']}",
                data=json.dumps(camera, sort_keys=True, allow_nan=False),
            )
        else:
            attributes = {"fovy": str(camera["fovy_deg"])}
            pose = rigid_transform(camera["T_body_camera"])
        ET.SubElement(parent, "camera", name=camera["name"], **attributes, **pose_attributes(pose))


class CameraCapture:
    """Shared collection/evaluation image path; owns and closes its render contexts."""

    def __init__(self, robot, names, *, width, height):
        self.model, self.size = robot.model, (width, height)
        definitions = robot.scene.configuration.get("cameras", []) if robot.scene else []
        self.definitions = {c["name"]: c for c in definitions if c["name"] in names}
        for name in names:
            index = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_TEXT, f"alohamini.camera.{name}"
            )
            if index >= 0:
                begin = self.model.text_adr[index]
                end = begin + self.model.text_size[index] - 1
                frozen = json.loads(bytes(self.model.text_data[begin:end]))
                if name in self.definitions and self.definitions[name] != frozen:
                    raise ValueError(f"Camera definition differs from the recorded model: {name}")
                self.definitions[name] = frozen
        self.lenses = {
            name: Lens(c["intrinsics"]) for name, c in self.definitions.items() if "intrinsics" in c
        }
        self.renderers = {}
        self.object_geometry = {}
        for name in names:
            self.model.camera(name)
            if name in self.lenses:
                lens = self.lenses[name]
                if not np.allclose(
                    model_matrix(self.model, name, lens.size),
                    lens.rendering[0],
                    rtol=2e-6,
                    atol=2e-4,
                ):
                    raise ValueError(f"Compiled camera differs from its calibration: {name}")

    def metadata(self, name):
        lens = self.lenses.get(name)
        K = (
            resize_matrix(lens.K, lens.size, self.size)
            if lens
            else model_matrix(self.model, name, self.size)
        )
        return {
            "resolution": list(self.size),
            "K_opencv": K.tolist(),
            "D": lens.D.tolist() if lens else [0.0] * 5,
            "distortion_model": lens.kind if lens else "plumb_bob",
            "pixel_centres": "OpenCV integer coordinates; top-left pixel centre (0,0)",
            "optical_frame": "X right, Y down, Z forward",
            "calibration": self.definitions.get(name),
            "processing": "raw lens distortion, full frame; RGB uint8",
        }

    def object_bounds(self, data, name, body):
        """Project task geometry for a full-frame gate; occlusion stays in RGB."""
        import cv2

        if body not in self.object_geometry:
            root = self.model.body(body).id
            bodies = {root}
            for index in range(root + 1, self.model.nbody):
                if self.model.body_parentid[index] in bodies:
                    bodies.add(index)
            ids = [
                i
                for i in range(self.model.ngeom)
                if self.model.geom_bodyid[i] in bodies and self.model.geom_group[i] == 2
            ]
            if not ids:
                raise ValueError(f"Workspace camera needs visible task geometry: {body}")
            pieces = []
            for index in ids:
                mesh = self.model.geom_dataid[index]
                if self.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
                    begin = self.model.mesh_vertadr[mesh]
                    points = self.model.mesh_vert[begin : begin + self.model.mesh_vertnum[mesh]]
                else:
                    size = self.model.geom_size[index].copy()
                    kind = int(self.model.geom_type[index])
                    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                        size[:] = size[0]
                    elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                        size = np.array(
                            [
                                size[0],
                                size[0],
                                size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0),
                            ]
                        )
                    elif kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                        raise ValueError(f"Unsupported workspace task geometry: {body}")
                    points = (
                        np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
                        * size
                    )
                pieces.append((index, points))
            self.object_geometry[body] = pieces
        points = np.concatenate(
            [
                vertices @ data.geom(index).xmat.reshape(3, 3).T + data.geom(index).xpos
                for index, vertices in self.object_geometry[body]
            ]
        )
        camera = data.camera(name)
        rotation = camera.xmat.reshape(3, 3) @ OPTICAL_TO_OPENGL[:3, :3]
        optical = (points - camera.xpos) @ rotation
        metadata = self.metadata(name)
        K, D = np.array(metadata["K_opencv"]), np.array(metadata["D"])
        project = (
            cv2.fisheye.projectPoints
            if metadata["distortion_model"] == "equidistant"
            else cv2.projectPoints
        )
        uv = project(optical.reshape(-1, 1, 3).astype(float), np.zeros(3), np.zeros(3), K, D)[0]
        low, high = uv.reshape(-1, 2).min(axis=0), uv.reshape(-1, 2).max(axis=0)
        passed = bool(
            np.isfinite(uv).all()
            and (optical[:, 2] > 0).all()
            and (low >= -0.5).all()
            and (high < np.array(self.size) - 0.5).all()
        )
        return np.r_[low, high], passed

    def image(self, data, name, options):
        lens = self.lenses.get(name)
        size = lens.size if lens else self.size
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(
                self.model,
                width=size[0],
                height=size[1],
                max_geom=max(10000, self.model.ngeom + 1000),
            )
        renderer = self.renderers[size]
        renderer.update_scene(data, camera=name, scene_option=options)
        rgb = renderer.render()
        if lens:
            rgb = lens.raw_image(rgb)
        if size != self.size:
            from PIL import Image

            rgb = np.asarray(Image.fromarray(rgb).resize(self.size, Image.Resampling.BILINEAR))
        return rgb

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
