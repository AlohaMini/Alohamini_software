"""Source-preserving component ownership for kinematic assembly inspection.

A reference configuration binds geometry to its new rigid owner. It is not a
mass redistribution or a qualification of the retained legacy collisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from .collision import components, visual_sources
from .frames import points_in_frame
from .meshes import binary_msh, binary_stl
from .model import pose_attributes
from .organization import description_hashes
from .shapes import Primitive


def bridge_joints(description, source, owner):
    """Return active joints on the unique tree path between two rigid links."""
    parents = {joint.child: joint for joint in description.joints.values()}

    def ancestors(link):
        result = {link: []}
        path = []
        while link in parents:
            joint = parents[link]
            path = [*path, joint]
            link = joint.parent
            result[link] = path
        return result

    first, second = ancestors(source), ancestors(owner)
    common = next(link for link in first if link in second)
    return [
        joint for joint in first[common] + list(reversed(second[common])) if joint.kind != "fixed"
    ]


def assembly_inputs(robot, assembly):
    """Partition complete source visuals and reparent without changing triangles.

    All components of each selected visual must be declared exactly once. The
    original source bytes, scale and visual pose are bound into the manifest.
    Relative transforms are derived from FK at explicit path-joint positions;
    arbitrary mesh offsets and implicit reference angles are not accepted.
    """
    if "rigid_assembly_fixture" in robot.manifest:
        raise ValueError("Rigid ownership inspection requires an unmodified robot assembly")
    if "cad_component_fixture" in robot.manifest and not robot.manifest.get(
        "cad_component_bindings"
    ):
        raise ValueError("Scoped collision components require explicit source bindings")
    path = Path(assembly).expanduser().resolve()
    original = path.read_bytes()
    definition = json.loads(original)
    description = robot.description
    if (
        not isinstance(definition, dict)
        or definition.get("schema_version") != 1
        or definition.get("robot_model") != description.robot_model.model_id
        or definition.get("units") != "m"
        or definition.get("source_description_sha256") != description_hashes(description)
        or not isinstance(definition.get("sources"), list)
        or not definition["sources"]
    ):
        raise ValueError("Rigid assembly requires the current description, units and sources")
    root = ET.fromstring(robot.xml)
    asset = root.find("asset")
    assets, meshes = dict(robot.assets), dict(robot.meshes)
    geometry, manifest = copy.deepcopy(robot.geometry), copy.deepcopy(robot.manifest)
    prepared, records, selected_sources, old_meshes = {}, [], set(), set()
    ownership, envelopes = {}, []
    for row in definition["sources"]:
        if not isinstance(row, dict):
            raise ValueError("Rigid assembly sources must be objects")
        link, index = row.get("link"), row.get("visual_index")
        if link not in description.links or type(index) is not int:
            raise ValueError("Rigid assembly needs a named source link and integer visual index")
        source_key = (link, index)
        if source_key in selected_sources:
            raise ValueError("A source visual cannot be assigned twice")
        selected_sources.add(source_key)
        sources = visual_sources(description, link)
        if not 0 <= index < len(sources) or row.get("source_visual") != sources[index]:
            raise ValueError("Rigid assembly source visual, pose, scale or checksum changed")
        source = sources[index]
        source_path = description.robot_model.directory / source["path"]
        contents = source_path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != source["sha256"]:
            raise ValueError("Rigid assembly source mesh changed during import")
        if source["sha256"] not in prepared:
            prepared[source["sha256"]] = components(binary_stl(contents))
        parts = prepared[source["sha256"]]
        assignments = row.get("components")
        if (
            not isinstance(assignments, list)
            or any(not isinstance(part, dict) for part in assignments)
            or any(type(part.get("component_index")) is not int for part in assignments)
            or sorted(part["component_index"] for part in assignments) != list(range(len(parts)))
        ):
            raise ValueError("Every source visual component must have exactly one rigid owner")
        original_name = f"{link}__visual_{index}"
        old = root.find(f".//body[@name='{link}']/geom[@name='{original_name}']")
        if old is None or original_name not in geometry:
            raise ValueError("Cannot locate the original source visual geometry")
        old_meshes.add(old.get("mesh"))
        root.find(f".//body[@name='{link}']").remove(old)
        del geometry[original_name]
        mapped = []
        for assignment in sorted(assignments, key=lambda item: item["component_index"]):
            component, owner = assignment["component_index"], assignment.get("owner")
            if owner not in description.links:
                raise ValueError("Component owner must be a named URDF link")
            joints = bridge_joints(description, link, owner)
            reference = assignment.get("reference_positions")
            if not isinstance(reference, dict) or set(reference) != {j.name for j in joints}:
                raise ValueError("Explicitly specify every path joint reference position")
            if any(type(value) not in (int, float) for value in reference.values()):
                raise ValueError("Assembly reference coordinates must be numeric radians/metres")
            for joint in joints:
                joint.check_position(reference[joint.name])
                if (
                    joint.name in robot.profile.locked_positions
                    and reference[joint.name] != robot.profile.locked_positions[joint.name]
                ):
                    raise ValueError("Assembly reference conflicts with a locked joint")
            positions = {**robot.initial_positions, **reference}
            poses = description.forward(positions)
            T_owner_source = np.linalg.inv(poses[owner]) @ poses[link]
            pose = T_owner_source @ source["T_link_visual"]
            mesh = parts[component]
            # Coordinates and scale remain in the original mesh frame. Keeping
            # the transform on the geom avoids another float32 vertex rounding.
            data = mesh.msh_bytes()
            digest = hashlib.sha256(data).hexdigest()
            relative, virtual = f"assembly/{digest}", f"assembly_{digest}.msh"
            mesh_name = (
                "assembly_"
                + hashlib.sha256((digest + json.dumps(source["scale"])).encode()).hexdigest()
            )
            if relative not in meshes:
                meshes[relative] = mesh
                assets[virtual] = data
            if asset.find(f"mesh[@name='{mesh_name}']") is None:
                ET.SubElement(
                    asset,
                    "mesh",
                    name=mesh_name,
                    file=virtual,
                    scale=" ".join(map(str, source["scale"])),
                )
            name = f"{original_name}__component_{component}"
            attributes = dict(old.attrib)
            attributes.pop("quat", None)
            attributes.pop("pos", None)
            attributes.update(name=name, mesh=mesh_name, **pose_attributes(pose))
            ET.SubElement(root.find(f".//body[@name='{owner}']"), "geom", **attributes)
            geometry[name] = {
                "link": owner,
                "kind": "visual",
                "T_link_geometry": pose,
                "shape": "mesh",
                "source_name": old.get("name"),
                "path": relative,
                "scale": np.asarray(source["scale"]),
            }
            mapped.append(
                {
                    "component_index": component,
                    "owner": owner,
                    "geom": name,
                    "triangles": len(mesh.faces),
                    "vertices": len(mesh.vertices),
                    "reference_positions": reference,
                    "T_owner_geometry": pose.tolist(),
                    "label": assignment.get("label", name),
                }
            )
            ownership[(link, index, component)] = (owner, reference, T_owner_source)
            if "collision_envelope" in assignment:
                envelope = assignment["collision_envelope"]
                if (
                    not isinstance(envelope, dict)
                    or set(envelope) != {"primitives", "limitations"}
                    or not isinstance(envelope["primitives"], list)
                    or not envelope["primitives"]
                    or not isinstance(envelope["limitations"], list)
                    or not envelope["limitations"]
                    or any(
                        not isinstance(item, str) or not item.strip()
                        for item in envelope["limitations"]
                    )
                ):
                    raise ValueError(
                        "Inspection envelopes require primitives and explicit limitations"
                    )
                envelopes.append((link, index, component, envelope))
        records.append(
            {
                "link": link,
                "visual_index": index,
                "source_visual": source,
                "triangles": sum(len(part.faces) for part in parts),
                "components": mapped,
            }
        )
    collision_records, collision_geoms = [], {}
    envelope_identities = {(link, index, component) for link, index, component, _ in envelopes}
    for name, binding in robot.manifest.get("cad_component_bindings", {}).items():
        source_link = binding["source_link"]
        identity = (source_link, binding["visual_index"], binding["component_index"])
        if identity in envelope_identities:
            raise ValueError(
                "A component cannot use both reference hulls and an inspection envelope"
            )
        owner, reference, pose = ownership.get(identity, (source_link, {}, np.eye(4)))
        body = root.find(f".//body[@name='{source_link}']")
        node = body.find(f"geom[@name='{name}']") if body is not None else None
        if node is None or any(
            key in node.attrib
            for key in (
                "pos",
                "quat",
                "euler",
                "axisangle",
                "xyaxes",
                "zaxis",
                "fromto",
            )
        ):
            raise ValueError("Scoped collision geometry must retain its source link coordinates")
        mesh_node = asset.find(f"mesh[@name='{node.get('mesh')}']")
        if mesh_node is None or mesh_node.get("scale", "1 1 1") != "1 1 1":
            raise ValueError("Scoped collision meshes must be in unscaled source-link metres")
        contents = assets[mesh_node.get("file")]
        relative = "assembly-collision/" + hashlib.sha256(contents).hexdigest()
        meshes[relative] = binary_msh(contents)
        body.remove(node)
        node.attrib.update(pose_attributes(pose))
        root.find(f".//body[@name='{owner}']").append(node)
        geometry[name].update(
            link=owner,
            shape="mesh",
            path=relative,
            scale=np.ones(3),
            T_link_geometry=pose,
        )
        collision_geoms.setdefault(owner, []).append(name)
        collision_records.append(
            {
                **binding,
                "owner": owner,
                "geom": name,
                "reference_positions": reference,
                "T_owner_geometry": pose.tolist(),
                "triangles": len(meshes[relative].faces),
                "vertices": len(meshes[relative].vertices),
                "shape": "mesh",
            }
        )
    envelope_bindings, envelope_geoms = {}, {}
    for source_link, index, component, envelope in envelopes:
        owner, reference, T_owner_source = ownership[(source_link, index, component)]
        for number, record in enumerate(envelope["primitives"]):
            primitive = Primitive.from_record(record)
            pose = T_owner_source @ primitive.T_link_geometry
            name = f"assembly_envelope_{source_link}_{index}_{component}_{number}"
            if name in geometry or root.find(f".//geom[@name='{name}']") is not None:
                raise ValueError("Assembly envelope geometry name is already in use")
            size = _primitive_size(primitive)
            ET.SubElement(
                root.find(f".//body[@name='{owner}']"),
                "geom",
                name=name,
                type=primitive.kind,
                size=" ".join(format(float(x), ".17g") for x in size if x > 0),
                mass="0",
                group="4",
                **pose_attributes(pose),
            )
            geometry[name] = {
                "link": owner,
                "kind": "collision",
                "shape": primitive.kind,
                "parameters": primitive.parameters,
                "T_link_geometry": pose,
            }
            binding = {
                "source_link": source_link,
                "visual_index": index,
                "component_index": component,
            }
            envelope_bindings[name] = binding
            envelope_geoms.setdefault(owner, []).append(name)
            collision_records.append(
                {
                    **binding,
                    "owner": owner,
                    "geom": name,
                    "reference_positions": reference,
                    "T_owner_geometry": pose.tolist(),
                    "shape": primitive.kind,
                    "primitive": primitive.record(),
                    "qualification": "inspection only; material and contact equivalence unverified",
                    "limitations": envelope["limitations"],
                }
            )
    # Unreferenced source visual assets can otherwise dominate native compile
    # memory. Remove only the replaced assets no longer used by any XML node.
    used_meshes = {node.get("mesh") for node in root.iter() if node.get("mesh")}
    for mesh_name in old_meshes - used_meshes:
        node = asset.find(f"mesh[@name='{mesh_name}']")
        if node is not None:
            asset.remove(node)
    used_files = {node.get("file") for node in asset if node.get("file")}
    assets = {name: data for name, data in assets.items() if name in used_files}
    if path.read_bytes() != original:
        raise ValueError("Rigid assembly definition changed during import")
    if any(path.read_bytes() != data for path, data in description.source_bytes.items()):
        raise ValueError("Robot description changed during assembly construction")
    for record in records:
        current = visual_sources(description, record["link"])[record["visual_index"]]
        if current != record["source_visual"]:
            raise ValueError("Rigid assembly source changed during construction")
    manifest["rigid_assembly_fixture"] = {
        "definition_sha256": hashlib.sha256(original).hexdigest(),
        "definition": definition,
        "sources": records,
        "collision_components": collision_records,
        "frames": {
            "transform": "T_owner_geometry maps scaled geometry coordinates into the owner link",
            "translation_unit": "m",
            "visual": "original source mesh vertices followed by the original URDF scale",
            "collision": "convex vertices or explicit primitive poses in the source link; metres",
        },
        "scope": "kinematic rigid ownership; complete selected visual triangle coverage",
        "collision_policy": "original collisions retained with original owners; unqualified",
        "inertia_policy": "original URDF inertias retained; component mass distribution unknown",
        "dynamics_allowed": False,
    }
    if collision_geoms:
        manifest["cad_component_geoms"] = collision_geoms
    if envelope_bindings:
        manifest["assembly_envelope_bindings"] = envelope_bindings
        manifest["assembly_envelope_geoms"] = envelope_geoms
    return dict(
        xml=ET.tostring(root, encoding="unicode"),
        assets=assets,
        meshes=meshes,
        geometry=geometry,
        manifest=manifest,
    )


def _primitive_size(primitive):
    """MuJoCo half dimensions, padded to its native three-element size array."""
    parameters = primitive.parameters
    if primitive.kind == "box":
        return np.asarray(parameters["size"]) / 2
    if primitive.kind == "cylinder":
        return np.array([parameters["radius"], parameters["length"] / 2, 0.0])
    return np.array([parameters["radius"], 0.0, 0.0])


def _primitive_anchors(primitive):
    size = _primitive_size(primitive)
    if primitive.kind == "cylinder":
        size = size[[0, 0, 1]]
    elif primitive.kind == "sphere":
        size = np.repeat(size[0], 3)
    return np.vstack([np.zeros(3), np.diag(size), -np.diag(size)])


def audit_assembly(robot, *, samples=9):
    """Check every native vertex at references and discrete path-joint sweeps.

    Shared configurations execute mj_forward once for the complete assembly.
    Mounting fit, continuous clearance and inertial accuracy are not certified.
    """
    if type(samples) is not int or samples < 2:
        raise ValueError("Assembly inspection requires at least two sweep samples")
    fixture = robot.manifest.get("rigid_assembly_fixture")
    if fixture is None:
        raise ValueError("No rigid assembly inspection fixture")
    model, description = robot.model, robot.description
    work, configurations = [], {}
    for source in fixture["sources"]:
        work.extend(
            {
                **component,
                "source_link": source["link"],
                "kind": "visual",
                "T_source_geometry": source["source_visual"]["T_link_visual"],
            }
            for component in source["components"]
        )
    work.extend(
        {
            **component,
            "kind": "collision",
            "T_source_geometry": component.get("primitive", {}).get("T_link_geometry", np.eye(4)),
        }
        for component in fixture["collision_components"]
    )
    items = []
    for component in work:
        name, owner = component["geom"], component["owner"]
        geom_id = model.geom(name).id
        primitive = (
            Primitive.from_record(component["primitive"]) if "primitive" in component else None
        )
        if primitive is None:
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            native = model.mesh_vert[start : start + count].astype(float)
            entry = robot.geometry[name]
            mesh = robot.meshes[entry["path"]]
            points = mesh.vertices * entry["scale"]
            preserved = int(model.mesh_facenum[mesh_id]) == len(mesh.faces)
        else:
            points = native = _primitive_anchors(primitive)
            preserved = bool(
                model.geom_type[geom_id]
                == getattr(mujoco.mjtGeom, f"mjGEOM_{primitive.kind.upper()}")
                and np.allclose(
                    model.geom_size[geom_id], _primitive_size(primitive), rtol=0, atol=1e-14
                )
            )
        local = points_in_frame(component["T_owner_geometry"], points)
        reference = {**robot.initial_positions, **component["reference_positions"]}
        poses = description.forward(reference, robot.profile.T_world_root)
        initial = points_in_frame(poses[owner], local)
        original = points_in_frame(
            poses[component["source_link"]] @ component["T_source_geometry"],
            points,
        )
        configs = [reference]
        for joint in bridge_joints(description, component["source_link"], owner):
            if joint.name in robot.profile.locked_positions:
                continue
            if joint.lower is None or joint.upper is None:
                raise ValueError("Assembly sweep requires explicit joint position limits")
            configs.extend(
                {**reference, joint.name: float(q)}
                for q in np.linspace(
                    joint.lower,
                    joint.upper,
                    samples,
                )
            )
        row = {key: value for key, value in component.items() if key != "T_source_geometry"}
        row.update(
            configurations=len(configs),
            native_owner_matches=bool(
                model.geom_bodyid[geom_id] == model.body(owner).id,
            ),
            native_geometry_preserved=preserved,
            reference_preservation_error_m=float(np.linalg.norm(initial - original, axis=1).max()),
            max_native_vertex_error_m=0.0,
            max_reference_displacement_m=0.0,
        )
        if primitive is None:
            row["native_triangles_preserved"] = preserved
        item = dict(row=row, geom_id=geom_id, native=native, local=local, initial=initial)
        items.append(item)
        for positions in configs:
            key = json.dumps(positions, sort_keys=True)
            configuration = configurations.setdefault(key, dict(positions=positions, items=[]))
            configuration["items"].append(item)
    for configuration in configurations.values():
        positions = configuration["positions"]
        data = robot.data(positions)
        poses = description.forward(positions, robot.profile.T_world_root)
        for item in configuration["items"]:
            row, geom_id = item["row"], item["geom_id"]
            expected = points_in_frame(poses[row["owner"]], item["local"])
            actual = (
                item["native"] @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
            )
            row["max_native_vertex_error_m"] = max(
                row["max_native_vertex_error_m"],
                float(cKDTree(expected).query(actual)[0].max()),
                float(cKDTree(actual).query(expected)[0].max()),
            )
            row["max_reference_displacement_m"] = max(
                row["max_reference_displacement_m"],
                float(np.linalg.norm(expected - item["initial"], axis=1).max()),
            )
    rows = [item["row"] for item in items]
    for row in rows:
        row["passed"] = bool(
            row["native_owner_matches"]
            and row["native_geometry_preserved"]
            and row["reference_preservation_error_m"] <= 1e-10
            and row["max_native_vertex_error_m"] <= 1e-6
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "scope": "discrete FK and source preservation only; not mounting or contact qualification",
        "samples_per_path_joint": samples,
        "native_configurations": len(configurations),
        "components": rows,
        "definition_sha256": fixture["definition_sha256"],
        "dynamics_allowed": False,
    }
