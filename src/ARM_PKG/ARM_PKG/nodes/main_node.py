#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 기본 로깅 설정 (원하면 레벨 조절 가능)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_node")


class ArmMainNode(Node):
    """
    OPC UA → read_opcua_node → /arm/opcua_event
    를 받아서 내부 로직을 분기하고,
    필요한 명령을 /arm/go_mode_cmd 등으로 전달하는 메인 오케스트레이션 노드.
    """

    def __init__(self):
        super().__init__("arm_main_node")

        # 1) OPCUA 이벤트 공통 토픽 구독
        self.opcua_event_sub = self.create_subscription(
            String,
            "/opcua_read_event",   # read_opcua_node에서 publish하는 토픽
            self._on_opcua_event,
            10
        )

        # 2) ARM 동작 명령을 내려줄 토픽 (go_mode_node가 나중에 구독 예정)
        self.go_mode_pub = self.create_publisher(
            String,
            "/go_move_cmd",
            10
        )

        # (선택) 나중에 카메라 트리거용 토픽도 쓸 수 있음
        # self.camera_trigger_pub = self.create_publisher(
        #     String,
        #     "/arm/camera_trigger",
        #     10
        # )

        self.get_logger().info("arm_main_node 초기화 완료")

    # --------------------------------------------------
    # 📥 OPCUA 이벤트 수신 콜백
    # --------------------------------------------------
    def _on_opcua_event(self, msg: String):
        """
        read_opcua_node가 넘겨준 이벤트(JSON 문자열)를 파싱하고,
        name/value에 따라 로직을 분기한다.
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"잘못된 JSON 수신: {msg.data}")
            return

        name = data.get("name")
        value = data.get("value")

        self.get_logger().info(f"[OPCUA EVENT] name={name}, value={value}")

        # name 에 따라 로직 분기
        if name == "arm_go_move":
            self._handle_arm_go_move(value)


        else:
            self.get_logger().warn(f"알 수 없는 OPCUA 이벤트 name='{name}'")

    # --------------------------------------------------
    # ♻ 개별 이벤트 처리 함수들 (지금은 단순 로깅 + 패스스루)
    #    나중에 여기 로직만 고쳐서 확장하면 됨.
    # --------------------------------------------------
    def _handle_arm_go_move(self, value):
        """
        PLC/OPCUA에서 온 ARM 이동 명령 처리.
        예: value == 'go_home', 'mission_start' 등.
        지금은 일단 그대로 /go_move_cmd 로 전달만 한다.
        """

        # 1) Ready는 OPCUA 초기 인사 신호 → 여기서 바로 무시
        if isinstance(value, str) and value.strip() == "Ready":
            self.get_logger().info("[MAIN] arm_go_move: 초기 Ready 신호 수신 → 무시하고 종료")
            return

        self.get_logger().info(f"[MAIN] arm_go_move 처리: {value}")

        cmd_msg = String()
        cmd_msg.data = str(value)
        self.go_mode_pub.publish(cmd_msg)
        self.get_logger().info(f"[PUBLISH] /go_move_cmd → {cmd_msg.data}")


# --------------------------------------------------
# 🚀 main
# --------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ArmMainNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("arm_main_node 종료(Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
