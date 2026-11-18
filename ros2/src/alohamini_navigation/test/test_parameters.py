import json

import pytest
import yaml
from alohamini_navigation.parameters import navigation_parameters

from alohamini.model import get_robot_model


def settings(name):
    model = get_robot_model("alohamini2edu")
    return yaml.safe_load(model.asset_path(f"config/navigation/{name}.yaml").read_text())


def test_collision_polygon_conversion_preserves_geometry_and_trigger_threshold():
    original = settings("collision")
    before = yaml.safe_dump(original)
    humble = navigation_parameters(original, "collision", "humble")
    jazzy = navigation_parameters(original, "collision", "jazzy")
    assert humble == original
    for name in original["collision_monitor"]["ros__parameters"]["polygons"]:
        old = humble["collision_monitor"]["ros__parameters"][name]
        new = jazzy["collision_monitor"]["ros__parameters"][name]
        assert [v for point in json.loads(new["points"]) for v in point] == old["points"]
        for count in range(20):
            assert (count > old["max_points"]) == (count >= new["min_points"])
    assert yaml.safe_dump(original) == before


def test_navigation_uses_jazzy_plugin_identifiers_and_retains_custom_bt_libraries():
    original = settings("nav2")
    original["bt_navigator"]["ros__parameters"]["plugin_lib_names"].append("custom_bt_nodes")
    jazzy = navigation_parameters(original, "nav2", "jazzy")
    assert jazzy["planner_server"]["ros__parameters"]["GridBased"]["plugin"] == (
        "nav2_navfn_planner::NavfnPlanner"
    )
    assert jazzy["controller_server"]["ros__parameters"]["progress_checker_plugins"] == [
        "progress_checker"
    ]
    assert jazzy["bt_navigator"]["ros__parameters"]["plugin_lib_names"] == ["custom_bt_nodes"]
    with pytest.raises(ValueError, match="Unsupported"):
        navigation_parameters(original, "nav2", "unknown")
