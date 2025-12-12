#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MissionState(enum.Enum):
    IDLE = 0
    WAIT_DRIVER = 1
    WAIT_CAMERA = 2


class ArmGoMoveNode(Node):
    """
    /go_move_cmd 토픽을 받아서 실제 ARM 동작 시퀀스를 orchestration 하는 노드.

    - main_node → /go_move_cmd : high-level 명령 (go_home, mission_start 등)
    - 이 노드는:
        * driver_node 에 실제 이동 명령 publish (/arm/driver_cmd)
        * camera_vision_node 에 비전 명령 publish (/arm/camera_cmd)
        * driver/camera 결과를 수신해 상태를 갱신
        * 최종 미션 결과를 /arm/mission_result 로 publish
    """

    def __init__(self):
        super().__init__("go_move_node")

        # ------------------------------
        # 1) main_node에서 내려오는 명령 구독
        # ------------------------------
        self.cmd_sub = self.create_subscription(
            String,
            "/go_move_cmd",     # main_node 에서 publish
            self._on_cmd,
            10
        )

        # ------------------------------
        # 2) driver / camera 결과 구독
        # ------------------------------
        self.driver_result_sub = self.create_subscription(
            String,
            "/arm/driver_result",
            self._on_driver_result,
            10
        )

        self.camera_result_sub = self.create_subscription(
            String,
            "/arm/camera_result",
            self._on_camera_result,
            10
        )

        # ------------------------------
        # 3) driver / camera / opcua_write 로 보낼 명령 publisher
        # ------------------------------
        self.driver_cmd_pub = self.create_publisher(
            String,
            "/arm/driver_cmd",     # arm_driver_node 가 구독 예정
            10
        )

        self.camera_cmd_pub = self.create_publisher(
            String,
            "/arm/camera_cmd",     # camera_vision_node 가 구독 예정
            10
        )

        self.mission_result_pub = self.create_publisher(
            String,
            "/arm/mission_result", # write_opcua_node 에서 구독 예정
            10
        )

        # ------------------------------
        # 4) 내부 상태
        # ------------------------------
        self.state = MissionState.IDLE
        self.current_command = None      # 예: "go_home", "mission_start"
        self.get_logger().info("go_move_node 초기화 완료")

    # ==================================================
    # 📥 1. main_node → /go_move_cmd 수신 콜백
    # ==================================================
    def _on_cmd(self, msg: String):
        """
        main_node 에서 내려준 high-level 명령 처리.
        예:
          - "go_home"
          - "mission_start"
        """
        raw = msg.data
        self.get_logger().info(f"[GO_MOVE] 수신 명령(raw): {raw}")

        # value가 JSON일 수도, 그냥 문자열일 수도 있으니 둘 다 지원
        cmd = self._extract_command(raw)
        if not cmd:
            self.get_logger().warn(f"[GO_MOVE] 인식 불가 명령: {raw}")
            return

        # 이미 진행 중이면(미션 수행 중) 추가 명령을 막거나 큐잉할 수 있음
        # if self.state != MissionState.IDLE:
        #     self.get_logger().warn(
        #         f"[GO_MOVE] 현재 미션 진행 중(state={self.state.name}) → 새 명령 '{cmd}' 무시"
        #     )
        #     return

        self.current_command = cmd
        self.get_logger().info(f"[GO_MOVE] 파싱된 명령: {cmd}")

        # 명령 종류에 따른 분기
        if cmd == "go_home":
            self._start_go_home_mission()
        elif cmd == "mission_start":
            self._start_mission_pick_sequence()
        else:
            self.get_logger().warn(f"[GO_MOVE] 지원하지 않는 명령: {cmd}")

    # --------------------------------------------------
    # 명령 문자열 / JSON 파싱 유틸
    # --------------------------------------------------
    def _extract_command(self, raw: str):
        """
        - raw == 'go_home'
        - raw == '{"move_command": "go_home"}'
        같은 경우 모두에서 최종 cmd 문자열('go_home')을 추출한다.
        """
        raw = raw.strip()
        # JSON 시도
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # OPCUA 쪽에서 {"move_command": "..."} 형태로 줄 가능성 고려
                if "move_command" in data:
                    return str(data["move_command"])
                elif "cmd" in data:
                    return str(data["cmd"])
        except Exception:
            # JSON 아니면 그냥 문자열로 처리
            pass

        # 그냥 평범한 문자열 명령이라고 가정
        return raw if raw else None

    # ==================================================
    # 2. go_home 미션 시작
    # ==================================================
    def _start_go_home_mission(self):
        """
        go_home 명령 수신 시:
          - driver_node 에 'go_home' 명령 전송
          - driver_result 를 기다리며 상태 WAIT_DRIVER 로 전환
        """
        self.get_logger().info("[GO_MOVE] go_home 미션 시작")

        cmd_payload = {
            "action": "go_home"
        }
        msg = String()
        msg.data = json.dumps(cmd_payload, ensure_ascii=False)
        self.driver_cmd_pub.publish(msg)

        self.state = MissionState.WAIT_DRIVER
        self.get_logger().info("[GO_MOVE] driver 결과 대기 상태로 진입 (go_home)")

    # ==================================================
    # 3. mission_start (픽업 + 비전) 시퀀스 시작
    # ==================================================
    def _start_mission_pick_sequence(self):
        """
        mission_start 명령 수신 시:
          1) driver_node 에 'move_to_pick_pose' 명령
          2) driver 결과가 OK면 camera_vision_node 에 비전 명령
          3) camera 결과에 따라 최종 미션 결과를 결정
        """
        self.get_logger().info("[GO_MOVE] mission_start 시퀀스 시작")

        cmd_payload = {
            "action": "move_to_pick_pose"
        }
        msg = String()
        msg.data = json.dumps(cmd_payload, ensure_ascii=False)
        self.driver_cmd_pub.publish(msg)

        self.state = MissionState.WAIT_DRIVER
        self.get_logger().info("[GO_MOVE] driver 결과 대기 상태로 진입 (mission_start)")

    # ==================================================
    # 📥 4. driver_result 콜백
    # ==================================================
    def _on_driver_result(self, msg: String):
        """
        arm_driver_node 가 publish 하는 결과를 처리.
        예:
          msg.data == '{"status": "success", "detail": "..."}'
        """
        raw = msg.data
        self.get_logger().info(f"[GO_MOVE] driver_result 수신: {raw}")

        try:
            data = json.loads(raw)
        except Exception:
            data = {"status": raw}

        status = str(data.get("status", "")).lower()

        if self.state != MissionState.WAIT_DRIVER:
            self.get_logger().warn(
                f"[GO_MOVE] driver_result 수신했지만 state={self.state.name}, 무시"
            )
            return

        # go_home or mission_start 에 따라 후속 동작 분기
        if self.current_command == "go_home":
            # go_home 은 driver 결과만으로 미션 종료
            self._finish_mission_with_driver_result(status, data)

        elif self.current_command == "mission_start":
            # mission_start 의 첫 단계: 픽업 포즈까지 이동
            if status == "success":
                # 다음 단계: 카메라/AI 검사 시작
                self._start_camera_inspection()
            else:
                # 이동 실패 → 전체 미션 실패 처리
                self._publish_mission_result(
                    success=False,
                    reason="driver_failed_before_camera",
                    extra=data
                )
                self._reset_state()

        else:
            self.get_logger().warn(
                f"[GO_MOVE] driver_result 처리할 current_command가 없음: {self.current_command}"
            )

    def _finish_mission_with_driver_result(self, status: str, data: dict):
        """
        go_home 처럼 driver 결과만으로 미션을 끝낼 때 사용하는 헬퍼.
        """
        success = (status == "success")
        reason = "driver_success" if success else "driver_failed"

        self._publish_mission_result(
            success=success,
            reason=reason,
            extra=data
        )
        self._reset_state()

    # ==================================================
    # 5. 카메라/AI 검사 시작
    # ==================================================
    def _start_camera_inspection(self):
        """
        mission_start 시퀀스에서:
          - driver가 픽업 포즈로 성공적으로 이동한 뒤
          - camera_vision_node 에 비전 검사를 요청
        """
        self.get_logger().info("[GO_MOVE] driver 성공 → 카메라 비전 검사 요청")

        cmd_payload = {
            "action": "inspect_pick_zone"
        }
        msg = String()
        msg.data = json.dumps(cmd_payload, ensure_ascii=False)
        self.camera_cmd_pub.publish(msg)

        self.state = MissionState.WAIT_CAMERA

    # ==================================================
    # 📥 6. camera_result 콜백
    # ==================================================
    def _on_camera_result(self, msg: String):
        """
        camera_vision_node 가 publish 하는 AI/비전 결과 처리.
        예:
          msg.data == '{"result": "ok", "module_type": "ESP32", "confidence": 0.98}'
        """
        raw = msg.data
        self.get_logger().info(f"[GO_MOVE] camera_result 수신: {raw}")

        if self.state != MissionState.WAIT_CAMERA:
            self.get_logger().warn(
                f"[GO_MOVE] camera_result 수신했지만 state={self.state.name}, 무시"
            )
            return

        try:
            data = json.loads(raw)
        except Exception:
            data = {"result": raw}

        result = str(data.get("result", "")).lower()
        success = (result == "ok")

        # TODO: 필요시 module_type, confidence 등에 따라 더 복잡한 분기 가능
        self._publish_mission_result(
            success=success,
            reason="camera_ok" if success else "camera_ng",
            extra=data
        )
        self._reset_state()

    # ==================================================
    # 7. 미션 결과 OPCUA 쪽으로 전달
    # ==================================================
    def _publish_mission_result(self, success: bool, reason: str, extra: dict = None):
        """
        최종 미션 결과를 /arm/mission_result 로 publish.
        나중에 write_opcua_node 에서 이 토픽을 구독해서
        OPC UA Method / 변수로 변환해 PLC에 전달한다.
        """
        payload = {
            "command": self.current_command,
            "success": success,
            "reason": reason,
            "detail": extra or {}
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.mission_result_pub.publish(msg)

        self.get_logger().info(
            f"[GO_MOVE] 미션 결과 publish → /arm/mission_result: {msg.data}"
        )

    # ==================================================
    # 8. 상태 초기화
    # ==================================================
    def _reset_state(self):
        self.get_logger().info(
            f"[GO_MOVE] 미션 종료 → state를 IDLE로 리셋 (command={self.current_command})"
        )
        self.state = MissionState.IDLE
        self.current_command = None


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
