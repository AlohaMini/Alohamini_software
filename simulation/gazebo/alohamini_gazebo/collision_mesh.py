"""Render OBJ collision surfaces as STL with explicit triangle normals."""

import hashlib
import math
from pathlib import Path


def collision_stl(source: Path, directory: Path) -> Path:
    content = source.read_bytes()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (hashlib.sha256(content).hexdigest() + ".stl")
    if destination.is_file():
        return destination
    vertices, faces = [], []
    for line in content.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertex = tuple(map(float, fields[1:4]))
            if len(vertex) != 3 or not all(map(math.isfinite, vertex)):
                raise ValueError(f"Invalid collision vertex in {source}")
            vertices.append(vertex)
        elif fields[0] == "f":
            indices = [int(field.split("/")[0]) for field in fields[1:]]
            if len(indices) < 3 or any(index == 0 for index in indices):
                raise ValueError(f"Invalid collision face in {source}")
            indices = [index - 1 if index > 0 else len(vertices) + index for index in indices]
            faces.extend((indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1))
    output = ["solid collision"]
    triangles = 0
    for face in faces:
        if any(index < 0 or index >= len(vertices) for index in face):
            raise ValueError(f"Collision face index is out of bounds in {source}")
        a, b, c = (vertices[index] for index in face)
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        length = math.sqrt(sum(value * value for value in normal))
        if length == 0:
            continue
        output.append("facet normal " + " ".join(format(value / length, ".17g") for value in normal))
        output.append("outer loop")
        output.extend("vertex " + " ".join(format(value, ".17g") for value in vertex) for vertex in (a, b, c))
        output.extend(("endloop", "endfacet"))
        triangles += 1
    if not triangles:
        raise ValueError(f"Collision mesh has no nondegenerate triangles: {source}")
    output.append("endsolid collision")
    destination.write_text("\n".join(output) + "\n", encoding="ascii")
    return destination
