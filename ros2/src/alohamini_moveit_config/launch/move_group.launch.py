"""MoveIt execution component using externally owned state and controllers."""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from alohamini.model import get_robot_model
from alohamini.model.joint_spec import ModelJointSpecification


def _nodes(context):
    model = get_robot_model(LaunchConfiguration("robot_model").perform(context))
    package = Path(get_package_share_directory("alohamini_moveit_config"))
    moveit_config = (
        MoveItConfigsBuilder(model.model_id, package_name="alohamini_moveit_config")
        .robot_description(file_path=str(model.description_path("collision")))
        .robot_description_semantic(file_path=str(model.description_path("semantic")))
        .robot_description_kinematics()
        .joint_limits()
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml",
            moveit_manage_controllers=False,
        )
        .planning_scene_monitor(
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    moveit_config.robot_description = {"robot_description": model.description_xml("collision")}
    moveit_config.planning_pipelines["ompl"].update(
        yaml.safe_load((package / "config" / "ompl_groups.yaml").read_text())
    )
    specification = ModelJointSpecification.from_model(model)
    limits = moveit_config.joint_limits["robot_description_planning"]["joint_limits"]
    for name, joint in specification.joints.items():
        limits[name] = dict(
            limits.get(name, {}),
            has_position_limits=True,
            min_position=joint.lower + joint.command_inset,
            max_position=joint.upper - joint.command_inset,
        )
    move_group_parameters = [
        moveit_config.to_dict(),
        {
            "allow_trajectory_execution": True,
            "publish_robot_description_semantic": True,
            "monitor_dynamics": False,
        },
    ]

    return [
        # /joint_states, TF and controller actions are externally owned by
        # AlohaMini Bringup. This component never starts hardware transport.
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=move_group_parameters,
        ),
        GroupAction(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            actions=[
                Node(
                    package="rviz2",
                    executable="rviz2",
                    name="moveit_rviz",
                    output="log",
                    arguments=["-d", str(package / "config" / "moveit.rviz")],
                    parameters=[
                        moveit_config.robot_description,
                        moveit_config.robot_description_semantic,
                        moveit_config.robot_description_kinematics,
                        moveit_config.planning_pipelines,
                        moveit_config.joint_limits,
                    ],
                )
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_model", default_value="alohamini2pro"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            OpaqueFunction(function=_nodes),
        ]
    )
