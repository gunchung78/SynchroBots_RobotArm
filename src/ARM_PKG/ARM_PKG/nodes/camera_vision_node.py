#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ArmCameraVisionNode(Node):
    """
    카메라에서 프레임을 읽고 Vision 처리를 수행하는 노드.

    현재는 간단한 구조:
      - 카메라 프레임 캡처
      - (옵션) 이미지 처리
      - 결과를 /vision_result 토픽으로 publish

    나중에:
      - AI 모델 로딩
      - pick 좌표 계산
      - OPC-UA 송신
      - arm_main_node 와 연동 등 확장 가능
    """

    def __init__(self):
        super().__init__("arm_camera_vision_node")

        # Vision 결과 토픽 publisher
        self.vision_pub = self.create_publisher(
            String,
            "/vision_result",
            10
        )

        # Timer (카메라 주기적 캡처)
        self.timer = self.create_timer(0.1, self._on_timer)  # 10Hz

        # 카메라 초기화 (0번 기본 웹캠 사용)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.get_logger().error("카메라를 열 수 없습니다 (index=0)")
        else:
            self.get_logger().info("카메라 초기화 성공")

        self.get_logger().info("arm_camera_vision_node 초기화 완료")

    # ----------------------------------------------------
    # 🔄 0.1초마다 실행되는 Vision 처리 함수
    # ----------------------------------------------------
    def _on_timer(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("카메라 프레임 읽기 실패")
            return

        # -------------------------------
        # (예시) 간단한 Vision 처리
        # 실제 Vision 로직은 나중에 붙이면 됨
        # -------------------------------
        height, width, _ = frame.shape
        result_msg = {
            "status": "ok",
            "frame_size": [width, height]
        }

        # Publish
        ros_msg = String()
        ros_msg.data = str(result_msg)
        self.vision_pub.publish(ros_msg)

    # ----------------------------------------------------
    # 종료 처리
    # ----------------------------------------------------
    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmCameraVisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("arm_camera_vision_node 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
