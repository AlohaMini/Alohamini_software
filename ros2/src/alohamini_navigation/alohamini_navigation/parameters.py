"""Translate shared navigation settings to the installed ROS distribution."""

import copy
import json
import os


def navigation_parameters(settings, kind, distro=None):
    result = copy.deepcopy(settings)
    distro = distro or os.environ.get("ROS_DISTRO", "humble")
    if distro == "humble":
        return result
    if distro != "jazzy":
        raise ValueError(f"Unsupported ROS distribution: {distro}")
    if kind == "collision":
        parameters = result["collision_monitor"]["ros__parameters"]
        parameters["enable_stamped_cmd_vel"] = False
        for name in parameters.get("polygons", []):
            polygon = parameters[name]
            points = polygon.get("points")
            if isinstance(points, list):
                if len(points) % 2:
                    raise ValueError("Collision polygon coordinates must be x/y pairs")
                polygon["points"] = json.dumps(list(zip(points[::2], points[1::2], strict=True)))
            if "max_points" in polygon:
                polygon["min_points"] = polygon.pop("max_points") + 1
    elif kind == "nav2":

        def plugins(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "plugin" and isinstance(item, str) and item.startswith("nav2_"):
                        value[key] = item.replace("/", "::")
                    else:
                        plugins(item)
            elif isinstance(value, list):
                for item in value:
                    plugins(item)

        plugins(result)
        controller = result["controller_server"]["ros__parameters"]
        if "progress_checker_plugin" in controller:
            controller["progress_checker_plugins"] = [controller.pop("progress_checker_plugin")]
        navigator = result["bt_navigator"]["ros__parameters"]
        # Jazzy loads its standard BT nodes itself; retain additional libraries.
        libraries = [
            name for name in navigator.pop("plugin_lib_names", []) if not name.startswith("nav2_")
        ]
        if libraries:
            navigator["plugin_lib_names"] = libraries
    return result
