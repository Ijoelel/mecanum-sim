import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    RegisterEventHandler,
    LogInfo
)
from launch.event_handlers import OnProcessExit


def generate_launch_description():

    # Package paths
    pkg_share = get_package_share_directory('mecanum_ros2')
    gz_share = get_package_share_directory('ros_gz_sim')

    # Robot description
    xacro_file = os.path.join(
        pkg_share,
        'urdf',
        'base.xacro'
    )

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )

    # Arena world
    world_file = os.path.join(
        pkg_share,
        'worlds',
        'arena.world'
    )

    # Launch Gazebo Fortress
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gz_share,
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': f'-r {world_file}'
        }.items()
    )

    # Publish robot description and TF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description
            }
        ]
    )

    # Spawn robot into Gazebo
    spawn_robot = ExecuteProcess(
        cmd=[
            'bash', '-c',
            'until ign service -l 2>/dev/null | grep -qx "/world/default/create" '
            '&& ros2 topic info /robot_description 2>/dev/null | grep -q "Publisher count: 1"; '
            'do sleep 0.5; done; '
            'echo "[mecanum_ros2] Gazebo and robot_description ready"; '
            'exec ros2 run ros_gz_sim create '
            '-world default '
            '-topic /robot_description '
            '-name mecanum '
            '-z 0.001'
        ],
        output='screen'
    )

    # Broadcaster
    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'wheel_velocity_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '60',
            '--service-call-timeout',
            '60',
            '--switch-timeout',
            '60'
        ],
        output='screen'
    )

    # Controller Event
    spawn_controller_event = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                LogInfo(
                    msg='Robot spawn process finished. Starting controllers.'
                ),
                controller_spawner
            ]
        )
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_controller_event,
        spawn_robot
    ])