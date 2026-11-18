import math

import pytest
from alohamini_gazebo.collision_mesh import collision_stl


def test_collision_surface_keeps_vertices_and_has_unit_normals(tmp_path):
    source = tmp_path / "surface.obj"
    original = "v 0 0 0\nv 0.01 0 0\nv 0 0.02 0\nf -3 -2 -1\nf 1 1 1\n"
    source.write_text(original)
    directory = tmp_path / "converted"
    converted = collision_stl(source, directory)
    rows = [line.split() for line in converted.read_text().splitlines()]
    vertices = [list(map(float, row[1:])) for row in rows if row[0] == "vertex"]
    normals = [list(map(float, row[2:])) for row in rows if row[0] == "facet"]
    assert vertices == [[0, 0, 0], [0.01, 0, 0], [0, 0.02, 0]]
    assert normals == [[0, 0, 1]]
    assert all(math.isclose(sum(x * x for x in n), 1) for n in normals)
    assert source.read_text() == original
    assert collision_stl(source, directory) == converted


@pytest.mark.parametrize("face", ["f 1 1 1", "f 1 2 4", "f 0 2 3"])
def test_invalid_collision_surfaces_fail_before_gazebo_starts(tmp_path, face):
    source = tmp_path / "invalid.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n" + face + "\n")
    with pytest.raises(ValueError):
        collision_stl(source, tmp_path / "converted")
