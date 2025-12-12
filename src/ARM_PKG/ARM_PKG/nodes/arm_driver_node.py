#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# (나중에 실제 로봇 제어 붙일 때)
# from pymycobot import MyCobot320


class ArmDriverNode(Node):
    """
    ArmDriverNode

    - go_move_node 에서 내려오는 저수준 모션 명령을 구독해서
      실제 로봇(MyCobot 320)을 제어하는 노드의 뼈대.

    - 지금은 구조/토픽만 잡아놓고, 실제 모션 제어/에러 처리/피드백은
      나중에 단계적으로 채워넣을 예정.
    """

    def __init__(self):
        super().__init__("arm_driver_node")

        # 1) go_move_node → driver 로 내려오는 명령 구독
        #   예: "/arm/driver_cmd" 토픽에 JSON/문자열 형태 명령 전달
        self.cmd_sub = self.create_subscription(
            String,
            "/arm/driver_cmd",          # 나중에 go_move_node와 맞춰서 수정 가능
            self._on_driver_cmd,
            10,
        )

        # 2) (선택) 드라이버 상태/결과를 상위(go_move_node)로 올려줄 토픽
        #    예: "완료", "에러", "진행 중" 등
        self.state_pub = self.create_publisher(
            String,
            "/arm/driver_state",
            10,
        )

        # 3) (선택) 나중에 MyCobot 320 실제 연결용 핸들
        # self.mc = None
        # self._init_robot()

        self.get_logger().info("arm_driver_node 초기화 완료")

    # --------------------------------------------------
    # 🦾 로봇 초기화 (나중에 구현용 자리)
    # --------------------------------------------------
    def _init_robot(self):
        """
        MyCobot 320 실제 연결 / 초기 포즈 세팅 등을
        나중에 여기서 구현하면 됨.
        """
        # try:
        #     self.mc = MyCobot320('/dev/ttyUSB0', 115200)
        #     self.get_logger().info("MyCobot320 연결 성공")
        # except Exception as e:
        #     self.get_logger().error(f"MyCobot320 연결 실패: {e}")
        #     self.mc = None
        pass

    # --------------------------------------------------
    # 📥 go_move_node → driver 명령 수신 콜백
    # --------------------------------------------------
    def _on_driver_cmd(self, msg: String):
        """
        상위(go_move_node)에서 내려온 명령을 받아
        실제 로봇 제어 로직으로 연결하는 자리.

        예:
          - "go_home"
          - "move_joint: [...]"
          - "move_pose: {...}"
        등으로 프로토콜 정의 가능.
        """
        raw_cmd = msg.data
        self.get_logger().info(f"[DRIVER] 수신 명령(raw): {raw_cmd}")

        # TODO:
        #  1) raw_cmd 파싱 (JSON / 'go_home' / 'move_joint: ...' 등)
        #  2) 명령 종류에 따라 MyCobot 제어 함수 호출
        #  3) 수행 결과를 /arm/driver_state 로 publish

        # 예시로 상태만 찍어 보내는 더미 로직:
        state_msg = String()
        state_msg.data = f"EXECUTED(dummy): {raw_cmd}"
        self.state_pub.publish(state_msg)
        self.get_logger().info(f"[DRIVER] 상태 publish: {state_msg.data}")

    # --------------------------------------------------
    # 🔚 종료 처리 (필요 시)
    # --------------------------------------------------
    def destroy_node(self):
        # if self.mc is not None:
        #     # 로봇 연결 해제 등
        #     pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmDriverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("arm_driver_node 종료(Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
