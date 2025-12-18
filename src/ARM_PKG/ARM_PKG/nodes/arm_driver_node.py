#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import cv2
import numpy as np
from pymycobot import MyCobot320

class ArmDriverNode(Node):
    def __init__(self):
        super().__init__("arm_driver_node")

        self.PORT, self.BAUD = "/dev/ttyUSB0", 115200
        self.MOVEMENT_SPEED = 70
        self.PICK_Z_HEIGHT = 260
        self.GRIPPER_SPEED = 50
        self.GRIPPER_OPEN, self.GRIPPER_CLOSE = 85, 25
        self.GRIPPER_DELAY = 1.0

        # 비전 좌표 보정 (Pixel to MM)
        self.CAMERA_INDEX = 0
        self.TARGET_CENTER_U, self.TARGET_CENTER_V = 320, 180
        self.PIXEL_TO_MM_X, self.PIXEL_TO_MM_Y = 0.526, -0.698

        # 로봇 주요 포즈
        self.CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90]
        self.ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90]
        self.INTERMEDIATE_POSE = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
        self.BASE_PICK_COORDS = [-237.90, 20, 183.6, -174.98, 0, 0]
        self.GLOBAL_TARGET_COORDS = [-114, -195, 250, 177.71, 0.22, 0]
        self.GLOBAL_TARGET_TMP_COORDS = [-150.0, -224.4, 318.1, 176.26, 3.2, 3.02]

        # 빨간색 검출용 HSV
        self.LOWER_RED_HSV1 = np.array([0, 100, 100])
        self.UPPER_RED_HSV1 = np.array([15, 255, 255])
        self.LOWER_RED_HSV2 = np.array([155, 100, 100])
        self.UPPER_RED_HSV2 = np.array([179, 255, 255])

        # 미션 카운트 (Place 완료 후 OPC UA 통신 트리거용)
        self.execute_mission_count = 0
        self.load_object_count = 2

        # --- 하드웨어 연결 ---
        self.mc = MyCobot320(self.PORT, self.BAUD)
        self.mc.init_electric_gripper()
        self.cap = cv2.VideoCapture(self.CAMERA_INDEX, cv2.CAP_V4L2)

        # --- ROS2 통신 설정 ---
        self.cmd_sub = self.create_subscription(String, "/arm/driver_cmd", self._on_driver_cmd, 10)
        self.state_pub = self.create_publisher(String, "/arm/driver_state", 10) # 상태 보고용

        self.get_logger().info("✅ Arm Driver Node initialized with MyCobot320")

    def _wait_stop(self, delay=2.0):
        while self.mc.is_moving():
            time.sleep(0.2)
        time.sleep(delay)

    def _on_driver_cmd(self, msg: String):
        try:
            data = json.loads(msg.data)
            action = data.get("action")
            
            if action == "go_home":
                self._execute_go_home()
            elif action == "move_to_pick":
                pick_coord = data.get("pick_coord")
                self._execute_pick_and_place(pick_coord)
                
        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")

    def _execute_go_home(self):
        self.get_logger().info("🏠 Moving to Home Pose...")
        self.mc.send_coords(self.INTERMEDIATE_POSE, self.MOVEMENT_SPEED)
        self._wait_stop()
        self.mc.send_angles(self.CONVEYOR_CAPTURE_POSE, self.MOVEMENT_SPEED)
        self._wait_stop()
        self._publish_state("HOME_DONE")

    def _execute_pick_and_place(self, pick_pose):
        self.execute_mission_count += 1
        self.get_logger().info(f"🚀 Starting Pick & Place (Count: {self.execute_mission_count})")

        # 1. Pick Action (Safety -> Pick -> Close)
        for z_off in [50, 0]:
            p = list(pick_pose)
            p[2] += z_off
            self.mc.send_coords(p, self.MOVEMENT_SPEED - 20)
            self._wait_stop(0.5)

        self.mc.set_gripper_value(self.GRIPPER_CLOSE, self.GRIPPER_SPEED)
        time.sleep(self.GRIPPER_DELAY)

        # -------------------------- #
        # 2. Place Action (Vision-Guided)
        # -------------------------- #
        self.mc.send_angles(self.ROBOTARM_CAPTURE_POSE, self.MOVEMENT_SPEED)
        self._wait_stop()

        # 카메라 버퍼 비우기
        for _ in range(15):
            self.cap.read()

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("❌ Place frame capture failed")
            return

        # 빨간색 영역 검출 (find_red_center)
        center_u, center_v = self._find_red_center(frame)

        if center_u is None:
            self.get_logger().error("🔴 Red object not detected. Moving to safe pose.")
            self.mc.send_angles(self.ROBOTARM_CAPTURE_POSE, self.MOVEMENT_SPEED)
            self._wait_stop()
            return

        # 픽셀 오차 -> 로봇 이동량 변환
        delta_X_mm, delta_Y_mm = self._convert_pixel_to_robot_move(center_u, center_v)

        # 최종 목표 좌표 생성
        final_place_coords = list(self.GLOBAL_TARGET_COORDS)
        final_place_coords[0] += delta_X_mm
        final_place_coords[1] += delta_Y_mm
        final_place_coords[2] = self.PICK_Z_HEIGHT

        # [STEP 1] Place 구역 위 안전 포즈로 이동
        safe_place_tmp = list(self.GLOBAL_TARGET_TMP_COORDS)
        self.mc.send_coords(safe_place_tmp, self.MOVEMENT_SPEED - 20)
        self._wait_stop()

        # [STEP 2] 정밀 좌표 하강
        self.mc.send_coords(final_place_coords, self.MOVEMENT_SPEED - 30)
        self._wait_stop()

        # [STEP 3] 그리퍼 열기
        self.mc.set_gripper_value(self.GRIPPER_OPEN, self.GRIPPER_SPEED)
        self._wait_stop(1.0)

        # [STEP 4] 복귀
        self.mc.send_coords(safe_place_tmp, self.MOVEMENT_SPEED)
        self._wait_stop()

        # 결과 보고 (main_node/OPC UA 연동을 위함)
        status = "arm_place_completed" if self.execute_mission_count % self.load_object_count == 0 else "arm_place_single"
        self._publish_state(status)
        self.get_logger().info(f"🏁 Mission Finished: {status}")

    def _find_red_center(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_frame, self.LOWER_RED_HSV1, self.UPPER_RED_HSV1)
        mask2 = cv2.inRange(hsv_frame, self.LOWER_RED_HSV2, self.UPPER_RED_HSV2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 50:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None, None

    def _convert_pixel_to_robot_move(self, current_u, current_v):
        delta_u = current_u - self.TARGET_CENTER_U
        delta_v = current_v - self.TARGET_CENTER_V
        return -(delta_u * self.PIXEL_TO_MM_X), -(delta_v * self.PIXEL_TO_MM_Y)

    def _publish_state(self, status):
        msg = String()
        msg.data = json.dumps({"status": status})
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArmDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mc.close()
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()