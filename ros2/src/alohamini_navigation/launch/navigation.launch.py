"""AMCL owns map -> odom; SLAM is deliberately absent from this graph."""

from pathlib import Path

from alohamini_navigation.launching import include, model_config, runtime_defaults
from alohamini_navigation.maps import validate_map_revision
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    path = Path(value("map")).expanduser().resolve()
    if not path.is_file() or path.suffix not in (".yaml", ".yml"):
        raise ValueError("map must be an existing saved map YAML")
    validate_map_revision(path, value("robot_model"))
    params = value("nav2_params") or model_config(value("robot_model"), "nav2")
    return [
        include("alohamini_navigation", "runtime.launch.py"),
        include(
            "nav2_bringup",
            "localization_launch.py",
            map=str(path),
            params_file=params,
            use_sim_time=value("use_sim_time"),
            autostart="true",
            use_composition="False",
        ),
        include(
            "nav2_bringup",
            "navigation_launch.py",
            params_file=params,
            use_sim_time=value("use_sim_time"),
            autostart="true",
            use_composition="False",
        ),
    ]


def generate_launch_description():
    defaults = dict(runtime_defaults(), nav2_params="")
    return LaunchDescription(
        [
            DeclareLaunchArgument("map"),
            *[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
            OpaqueFunction(function=_nodes),
        ]
    )
