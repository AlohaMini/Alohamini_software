"""Primitive research scenes in SAPIEN/ManiSkill, with Z-axis cylinders."""

import numpy as np


def add_objects(scene, configuration, *, render=True):
    import sapien

    actors = {}
    for obj in configuration["objects"]:
        builder = scene.create_actor_builder()
        shape, size = obj["shape"], obj["size"]
        color = obj.get("color", [0.7, 0.7, 0.7, 1])
        friction = obj.get("friction", 1)
        material = sapien.physx.PhysxMaterial(friction, friction, 0)
        if shape == "box":
            builder.add_box_collision(half_size=np.array(size) / 2, material=material)
            if render:
                builder.add_box_visual(half_size=np.array(size) / 2, material=color)
        elif shape == "sphere":
            builder.add_sphere_collision(radius=size[0], material=material)
            if render:
                builder.add_sphere_visual(radius=size[0], material=color)
        else:
            pose = sapien.Pose(q=[2**-0.5, 0, -(2**-0.5), 0])
            builder.add_cylinder_collision(
                radius=size[0], half_length=size[1] / 2, pose=pose, material=material
            )
            if render:
                builder.add_cylinder_visual(
                    radius=size[0], half_length=size[1] / 2, pose=pose, material=color
                )
        builder.initial_pose = sapien.Pose(obj.get("T_world_object", np.eye(4)))
        mass = obj.get("mass_kg", 0)
        if mass > 0:
            if shape == "box":
                x, y, z = size
                inertia = mass / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
            elif shape == "sphere":
                inertia = np.full(3, 0.4 * mass * size[0] ** 2)
            else:
                r, length = size
                inertia = np.array(
                    [mass * (3 * r * r + length * length) / 12] * 2 + [mass * r * r / 2]
                )
            builder.set_mass_and_inertia(mass, sapien.Pose(), inertia)
            actor = builder.build(name=obj["name"])
        else:
            actor = builder.build_static(name=obj["name"])
        actors[obj["name"]] = actor
    return actors
