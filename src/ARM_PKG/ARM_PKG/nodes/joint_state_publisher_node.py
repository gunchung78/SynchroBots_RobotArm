#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pymycobot import MyCobot320


# 🔧 MyCobot 연결 설정 (환경에 맞게 수정)
MC_SERIAL_PORT = "/dev/ttyUSB0"   # VirtualBox에서 USB 패스스루 한 포트
MC_BAUDRATE   = 115200

# ⚠️ joint 이름은 URDF에 정의된 이름과 맞춰야 RViz / Web에서 잘 움직인다.
#   mycobot_320 URDF에서 joint 이름 확인해서 필요하면 수정해줘.
JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]


class MyCobotJointStatePublisher(Node):
    """
    MyCobot 320의 실제 관절 각도를 읽어서 /joint_states 로 퍼블리시하는 노드.

    - pymycobot.MyCobot320 으로 주기적으로 get_angles() 호출
    - 결과를 sensor_msgs/JointState 로 변환해서 publish
    """

    def __init__(self):
        super().__init__("mycobot_joint_state_publisher")

        # 1) 퍼블리셔 생성
        self.joint_pub = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        # 2) MyCobot 연결
        self.mc = None
        self._init_robot()

        # 3) 주기적으로 JointState 발행 (예: 20 Hz → 0.05초 간격)
        self.timer_period = 0.05  # seconds
        self.timer = self.create_timer(self.timer_period, self._publish_joint_state)

        self.get_logger().info("mycobot_joint_state_publisher 초기화 완료")

    # --------------------------------------------------
    # 🦾 로봇 초기화
    # --------------------------------------------------
    def _init_robot(self):
        try:
            self.mc = MyCobot320(MC_SERIAL_PORT, MC_BAUDRATE)
            self.get_logger().info(
                f"MyCobot320 연결 성공: port={MC_SERIAL_PORT}, baud={MC_BAUDRATE}"
            )
        except Exception as e:
            self.get_logger().error(f"MyCobot320 연결 실패: {e}")
            self.mc = None

    # --------------------------------------------------
    # ⏱ 타이머 콜백: JointState 발행
    # --------------------------------------------------
    def _publish_joint_state(self):
        # 로봇 연결 안 되어 있으면 아무 것도 안 함
        if self.mc is None:
            # 너무 시끄럽지 않게 주기적인 에러 로그는 피하고 싶으면 debug로 내려도 됨
            self.get_logger().debug("MyCobot 연결 없음 → /joint_states 발행 생략")
            return

        try:
            # get_angles() → [deg1, deg2, ..., deg6] 또는 None
            angles_deg = self.mc.get_angles()
        except Exception as e:
            self.get_logger().warn(f"get_angles() 호출 중 오류: {e}")
            return

        if not angles_deg or len(angles_deg) < len(JOINT_NAMES):
            self.get_logger().warn(f"잘못된 각도 데이터 수신: {angles_deg}")
            return

        # 도(degree) → 라디안 변환
        angles_rad = [math.radians(a) for a in angles_deg[: len(JOINT_NAMES)]]

        # JointState 메시지 생성
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = angles_rad
        # (velocity / effort 는 생략 가능, 필요하면 나중에 추가)

        self.joint_pub.publish(msg)
        # debug 수준으로만 찍어도 충분
        self.get_logger().debug(f"/joint_states 발행: {angles_rad}")

    # --------------------------------------------------
    # 🔚 종료 처리
    # --------------------------------------------------
    def destroy_node(self):
        if self.mc is not None:
            try:
                # pymycobot에 close 함수 있으면 호출, 없으면 pass
                self.mc.release_all_servos()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MyCobotJointStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("mycobot_joint_state_publisher 종료(Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
