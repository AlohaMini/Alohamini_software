from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco import contact_probe
from alohamini_mujoco import viewer as inspection
from alohamini_mujoco.control import DynamicFollower
from alohamini_mujoco.solid_geometry import convex_boundary


@dataclass
class InspectionRobot:
    xml: str
    assets: dict
    model: object
    geometry: dict
    manifest: dict
    description: object
    scene: object = None

    def data(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        return data


def fixture(monkeypatch):
    xml = """<mujoco><asset/><worldbody><body name="part" pos=".1 0 .2">
      <joint name="pivot" axis="0 0 1" range="-1 1"/>
      <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
      <geom name="visual" type="sphere" size=".01" pos=".04 0 0" group="2"/>
      <geom name="legacy" type="sphere" size=".02" pos=".04 0 0" group="3"/>
    </body></worldbody></mujoco>"""
    source = InspectionRobot(
        xml,
        {},
        mujoco.MjModel.from_xml_string(xml),
        {
            "visual": {"kind": "visual", "link": "part"},
            "legacy": {"kind": "collision", "link": "part"},
        },
        {},
        SimpleNamespace(links={"part": None}),
    )

    def load(description, directory):
        index = int(directory)
        vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        mesh = convex_boundary(vertices + [0.04 + index * 0.02, 0, 0])
        return {"part": [mesh]}, {
            "manifest_sha256": str(index),
            "definition": {
                "links": {
                    "part": {
                        "cad_geometry": {
                            "registration": {
                                "target": {"visual_index": 0, "component_index": index}
                            }
                        }
                    }
                }
            },
        }

    monkeypatch.setattr(contact_probe, "load_cad_components", load)
    return source


def test_component_assembly_keeps_sources_and_inertia_compiles_once_and_rejects_duplicates(
    monkeypatch,
):
    source = fixture(monkeypatch)
    calls, native_compile = [], contact_probe.compile_xml

    def compile_once(*args, **kwargs):
        calls.append(True)
        return native_compile(*args, **kwargs)

    monkeypatch.setattr(contact_probe, "compile_xml", compile_once)
    combined = contact_probe.attach_component_fixtures(source, [0, 1])
    assert len(calls) == 1
    assert combined.model.ngeom == source.model.ngeom + 2
    np.testing.assert_array_equal(combined.model.body_mass, source.model.body_mass)
    np.testing.assert_array_equal(combined.model.body_inertia, source.model.body_inertia)
    assert len(combined.manifest["cad_component_geoms"]["part"]) == 2
    assert len(combined.manifest["cad_component_fixture"]["components"]) == 2
    assert source.manifest == {}
    with pytest.raises(ValueError, match="cannot run dynamics"):
        DynamicFollower(combined)
    with pytest.raises(ValueError, match="already present"):
        contact_probe.attach_component_fixtures(source, [0, 0])
    assert len(calls) == 1


def test_model_builder_adds_components_before_its_only_native_compilation(monkeypatch):
    from alohamini_mujoco import model as builder

    calls, sentinel = [], object()
    mesh = convex_boundary(np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]))

    def load(description, directory):
        return {"left_Fixed_Jaw": [mesh]}, {
            "definition": {
                "links": {
                    "left_Fixed_Jaw": {
                        "cad_geometry": {
                            "registration": {
                                "target": {"visual_index": 0, "component_index": directory}
                            }
                        }
                    }
                }
            }
        }

    def compile_once(xml, **kwargs):
        calls.append(xml)
        assert 'name="cad_fixture0_left_Fixed_Jaw_0"' in xml
        assert 'name="cad_fixture1_left_Fixed_Jaw_0"' in xml
        return sentinel

    monkeypatch.setattr(contact_probe, "load_cad_components", load)
    monkeypatch.setattr(builder, "compile_xml", compile_once)
    robot = builder.build_model(component_bundles=[0, 1])
    assert robot.model is sentinel
    assert len(calls) == 1
    assert "cad_component_fixture" in robot.manifest


def test_component_window_uses_joint_sliders_without_physics_steps_and_resets(monkeypatch):
    source = fixture(monkeypatch)
    combined = contact_probe.attach_component_fixtures(source, [0, 1])
    snapshots = []
    watched = combined.model.geom(combined.manifest["cad_component_geoms"]["part"][0]).id

    class Window:
        def __init__(self, model, data, key_callback, **kwargs):
            self.model, self.data, self.callback = model, data, key_callback
            self.opt, self.cam = mujoco.MjvOption(), mujoco.MjvCamera()
            assert kwargs["show_right_ui"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def lock(self):
            return nullcontext()

        def is_running(self):
            return len(snapshots) < 4

        def set_texts(self, texts):
            assert "partial link coverage" in texts[3]

        def sync(self):
            snapshots.append(
                (
                    self.data.qpos.copy(),
                    self.data.geom_xpos[watched].copy(),
                    self.opt.geomgroup.copy(),
                    self.data.time,
                )
            )
            if len(snapshots) == 1:
                self.data.qpos[0] = 0.5
                self.data.qvel[0] = 3
                self.callback(296)  # overlay
            elif len(snapshots) == 2:
                self.callback(297)  # switch to source only
            elif len(snapshots) == 3:
                self.callback(298)  # reset

    def forbidden_step(*args, **kwargs):
        raise AssertionError("Kinematic component inspection must not step physics")

    monkeypatch.setattr(inspection, "check_display", lambda: None)
    monkeypatch.setattr(inspection.time, "sleep", lambda _: None)
    monkeypatch.setattr(inspection, "component_window", Window)
    monkeypatch.setattr(mujoco, "mj_step", forbidden_step)
    data = inspection.run_component_viewer(combined)
    assert [s[0][0] for s in snapshots] == [0, 0.5, 0.5, 0]
    assert all(s[3] == 0 for s in snapshots)
    assert not np.allclose(snapshots[0][1], snapshots[1][1])
    np.testing.assert_allclose(snapshots[0][1], snapshots[3][1], atol=1e-14)
    assert [s[2][2] for s in snapshots] == [0, 1, 1, 1]
    assert [s[2][4] for s in snapshots] == [1, 1, 0, 0]
    assert not any(s[2][3] for s in snapshots)  # legacy collisions stay hidden
    assert data.qvel[0] == 0
    assert source.model.geom_group[source.model.geom("legacy").id] == 3
    assert source.data().qpos[0] == 0
    with pytest.raises(ValueError, match="covered"):
        inspection.run_component_viewer(combined, links=["unknown"])


def test_component_window_joins_native_cleanup_and_propagates_launch_errors(monkeypatch):
    import threading

    from mujoco import viewer

    closed, destroyed = threading.Event(), threading.Event()

    class Handle:
        def close(self):
            closed.set()

    def native(model, data, *, handle_return, **kwargs):
        handle_return.put(Handle())
        assert closed.wait(2)
        destroyed.set()

    monkeypatch.setattr(inspection.sys, "platform", "linux")
    monkeypatch.setattr(viewer, "_launch_internal", native)
    with inspection.component_window(None, None):
        assert not destroyed.is_set()
    assert destroyed.is_set()
    closed.clear()
    destroyed.clear()
    with pytest.raises(KeyboardInterrupt):
        with inspection.component_window(None, None):
            raise KeyboardInterrupt
    assert destroyed.is_set()

    def failed(*args, **kwargs):
        raise ValueError("Native launch failed")

    monkeypatch.setattr(viewer, "_launch_internal", failed)
    with pytest.raises(RuntimeError, match="Cannot launch") as error:
        with inspection.component_window(None, None):
            pass
    assert isinstance(error.value.__cause__, ValueError)

    def cleanup_failed(model, data, *, handle_return, **kwargs):
        handle_return.put(Handle())
        assert closed.wait(2)
        raise ValueError("Native cleanup failed")

    monkeypatch.setattr(viewer, "_launch_internal", cleanup_failed)
    with pytest.raises(RuntimeError, match="viewer failed"):
        with inspection.component_window(None, None):
            pass
