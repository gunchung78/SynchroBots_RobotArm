#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ArmGoMoveNode(Node):
    """
    ✅ 토픽 기반이지만 Service처럼 동작하는 go_move_node

    - /go_move_cmd : 단건 요청
    - 1회 처리 후 즉시 lock
    - 작업 완료 시 unlock
    """

    def __init__(self):
        super().__init__("go_move_node")

        # -------------------------
        # 단건 처리용 락
        # -------------------------
        self.cmd_locked = False
        self.current_cmd = None

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

        self.driver_cmd_pub = self.create_publisher(
            String,
            "/arm/driver_cmd",
            10
        )

        self.get_logger().info("✅ go_move_node(Service-like MODE) 초기화 완료")

    # ==================================================
    # 1) /go_move_cmd 수신 (🔥 단건 처리)
    # ==================================================
    def _on_cmd(self, msg: String):
        raw = msg.data
        self.get_logger().info(f"[GO_MOVE] 수신(/go_move_cmd): {raw}")

        # 🔒 이미 처리 중이면 무시
        if self.cmd_locked:
            self.get_logger().warn("[GO_MOVE] cmd locked → 무시")
            return

        cmd = self._parse_cmd_json(raw)
        if not cmd:
            self.get_logger().warn("[GO_MOVE] cmd 파싱 실패 → 무시")
            return

        # 🔒 즉시 lock
        self.cmd_locked = True
        self.current_cmd = cmd

        # -------------------------
        # 명령 분기
        # -------------------------
        if cmd == "mission_start":
            self.get_logger().info("[GO_MOVE] mission_start → camera 호출")
            self._send_camera_action("inspect_pick_zone")

        elif cmd == "go_home":
            self.get_logger().info("[GO_MOVE] go_home → driver 전달")
            self._send_driver_action("go_home")
            self._unlock_cmd_delayed(3.0)   # 테스트 단계용 자동 unlock

        else:
            self.get_logger().warn(f"[GO_MOVE] 지원하지 않는 cmd: {cmd}")
            self._unlock_cmd()

    def _parse_cmd_json(self, raw: str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "move_command" in data:
                return str(data["move_command"]).strip()
        except Exception:
            return None
        return None

    # ==================================================
    # 2) camera 명령
    # ==================================================
    def _send_camera_action(self, action: str):
        payload = {"action": action}
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.camera_cmd_pub.publish(out)
        self.get_logger().info(f"[GO_MOVE] → /arm/camera_cmd: {out.data}")

    # ==================================================
    # 3) camera_result 수신
    # ==================================================
    def _on_camera_result(self, msg: String):
        if self.current_cmd != "mission_start":
            return

        raw = msg.data
        self.get_logger().info(f"[GO_MOVE] camera_result 수신: {raw}")

        data = self._safe_json(raw)
        status = data.get("status", "")
        pick_coord = data.get("pick_coord", None)
        final_rz = data.get("final_rz", None)

        if status != "success":
            self.get_logger().warn(
                f"[GO_MOVE] ❌ camera 실패: {data.get('reason')}"
            )
            self._unlock_cmd()
            return

        if not (isinstance(pick_coord, list) and len(pick_coord) == 6):
            self.get_logger().warn(
                f"[GO_MOVE] pick_coord 형식 이상: {pick_coord}"
            )
            self._unlock_cmd()
            return

        # ✅ move_to_pick 1회 전송
        payload = {
            "action": "move_to_pick",
            "pick_coord": pick_coord
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.driver_cmd_pub.publish(out)

        self.get_logger().info(
            f"[GO_MOVE] ✅ move_to_pick 전달 완료\n"
            f"  pick_coord = {pick_coord}\n"
            f"  final_rz   = {final_rz}"
        )

        # ✅ mission 완료 → unlock
        self._unlock_cmd()

    # ==================================================
    # unlock 로직
    # ==================================================
    def _unlock_cmd(self):
        self.get_logger().info("[GO_MOVE] cmd unlock")
        self.cmd_locked = False
        self.current_cmd = None

    def _unlock_cmd_delayed(self, delay_sec: float):
        self.create_timer(delay_sec, self._unlock_cmd)

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
