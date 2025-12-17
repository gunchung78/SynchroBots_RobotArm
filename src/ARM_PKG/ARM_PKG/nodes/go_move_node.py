#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ArmGoMoveNode(Node):
    """
    ✅ 테스트 단계 전용 go_move_node

    mission_start 수신 시:
      → 바로 camera_vision_node 트리거
      → camera_result(좌표)를 받아 로그로 확인

    ❌ driver 관련 로직 없음
    ❌ 로봇 제어 없음
    """

    def __init__(self):
        super().__init__("go_move_node")

        # -------------------------
        # Subscribers
        # -------------------------
        self.cmd_sub = self.create_subscription(
            String,
            "/go_move_cmd",
            self._on_cmd,
            10
        )

        self.camera_result_sub = self.create_subscription(
            String,
            "/arm/camera_result",
            self._on_camera_result,
            10
        )

        # -------------------------
        # Publishers
        # -------------------------
        self.camera_cmd_pub = self.create_publisher(
            String,
            "/arm/camera_cmd",
            10
        )

        self.get_logger().info("✅ go_move_node(TEST MODE) 초기화 완료")

    # ==================================================
    # 1) /go_move_cmd 수신
    # ==================================================
    def _on_cmd(self, msg: String):
        raw = msg.data
        self.get_logger().info(f"[GO_MOVE] 수신(/go_move_cmd): {raw}")

        cmd = self._parse_cmd_json(raw)
        if not cmd:
            self.get_logger().warn("[GO_MOVE] cmd 파싱 실패 → 무시")
            return

        if cmd == "mission_start":
            self.get_logger().info("[GO_MOVE] mission_start → camera 바로 호출")
            self._send_camera_action("inspect_pick_zone")
        else:
            self.get_logger().warn(f"[GO_MOVE] 지원하지 않는 cmd: {cmd}")

    def _parse_cmd_json(self, raw: str):
        """
        입력 포맷:
          {"move_command":"mission_start"}
        """
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "move_command" in data:
                return str(data["move_command"]).strip()
        except Exception:
            return None
        return None

    # ==================================================
    # 2) camera로 명령 보내기
    # ==================================================
    def _send_camera_action(self, action: str):
        payload = {"action": action}
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.camera_cmd_pub.publish(out)

        self.get_logger().info(f"[GO_MOVE] → /arm/camera_cmd: {out.data}")

    # ==================================================
    # 3) camera_result 수신 (좌표 확인용)
    # ==================================================
    def _on_camera_result(self, msg: String):
        raw = msg.data
        self.get_logger().info(f"[GO_MOVE] camera_result 수신: {raw}")

        data = self._safe_json(raw)

        status = data.get("status", "")
        pick_coord = data.get("pick_coord", None)
        final_rz = data.get("final_rz", None)

        if status == "success":
            self.get_logger().info(
                f"[GO_MOVE] ✅ 좌표 수신 성공\n"
                f"  pick_coord = {pick_coord}\n"
                f"  final_rz   = {final_rz}"
            )
        else:
            self.get_logger().warn(
                f"[GO_MOVE] ❌ camera 실패: {data.get('reason')}"
            )

    def _safe_json(self, raw: str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {"value": v}
        except Exception:
            return {"value": raw}


def main(args=None):
    rclpy.init(args=args)
    node = ArmGoMoveNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("go_move_node 종료(Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
