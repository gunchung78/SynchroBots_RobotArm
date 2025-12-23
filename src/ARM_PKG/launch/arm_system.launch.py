from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        


        # -------------------------
        # 2) Main Node
        # -------------------------
        Node(
            package='ARM_PKG',
            executable='arm_main_node',
            name='arm_main_node',
            output='screen'
        ),

        # -------------------------
        # 3) Go Move Node
        # -------------------------
        Node(
            package='ARM_PKG',
            executable='go_move_node',
            name='go_move_node',
            output='screen'
        ),
        Node(
            package="ARM_PKG",
            executable="arm_driver_node",
            name="arm_driver_node",
            output="screen",
        ),

        Node(
            package="ARM_PKG",
            executable="camera_vision_node",
            name="camera_vision_node",
            output="screen",
            parameters=[
                {"camera_index": 0},
                {"use_dshow": True},        # Windows면 True
                {"flush_frames": 10},
                {"enable_ai": True},
            ],
        ),
        Node(
            package="ARM_PKG",
            executable="write_opcua_node",
            name="write_opcua_node",
            output="screen",
            parameters=[{"enable_opcua": True}],
        ),

        # -------------------------
        # 1) OPCUA Read Node
        # -------------------------
        Node(
            package='ARM_PKG',
            executable='read_opcua_node',
            name='read_opcua_node',
            output='screen'
        ),
    ])
