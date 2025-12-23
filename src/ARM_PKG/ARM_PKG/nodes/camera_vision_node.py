#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import base64
import logging
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory
import os

# torch 관련은 환경에 따라 없을 수 있으니 안전 import
try:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms, models
    TORCH_OK = True
except Exception:
    TORCH_OK = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera_vision_node")


# =========================
# 기존 담당자 모델 구조(유지)
# =========================
class ResNetMultiTask(nn.Module):
    """Rz 추론을 위한 Multi-Task ResNet50 모델 구조"""
    def __init__(self, num_classes=17):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        common_fc = lambda out: nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out),
        )
        self.cls_head = common_fc(num_classes)
        self.res_head = common_fc(1)

    def forward(self, x):
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.cls_head(x), self.res_head(x)

class ArmCameraVisionNode(Node):
    """
    ✅ 여기서는 로봇암을 움직이지 않는다.

    1) /arm/camera_cmd 수신 → 카메라 캡쳐 + AI/비전( JOINT_DETECTION ) 수행
    2) AI 결과(분류/신뢰도/좌표/이미지 등) → /arm/ai_result 로 publish (write_opcua_node가 처리)
    3) 좌표(pick_coord, final_rz) → /arm/camera_result 로 publish (go_move_node가 처리)
    """

    def __init__(self):
        super().__init__("camera_vision_node")

        # -----------------------------
        # ROS I/O
        # -----------------------------
        self.cmd_sub = self.create_subscription(String, "/arm/camera_cmd", self._on_cmd, 10)
        self.coord_pub = self.create_publisher(String, "/arm/camera_result", 10)  # go_move_node용(좌표만)
        self.ai_pub = self.create_publisher(String, "/arm/ai_result", 10)         # write_opcua_node용(AI로그)

        # -----------------------------
        # 최소 파라미터(필수만)
        # -----------------------------
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("use_dshow", True)  # Windows면 True 권장
        self.declare_parameter("flush_frames", 10)

        self.declare_parameter("enable_ai", True)
        self.declare_parameter("model_cls_path", "best_trck_obj_cls_model.pth")
        self.declare_parameter("model_rz_path", "best_trck_coords_tracking_model.pth")

        raw_cls = self.get_parameter("model_cls_path").get_parameter_value().string_value
        raw_rz  = self.get_parameter("model_rz_path").get_parameter_value().string_value

        self.model_cls_path = self._resolve_model_path(raw_cls)
        self.model_rz_path  = self._resolve_model_path(raw_rz)

        self.get_logger().info(f"[CAM] model_cls_path = {self.model_cls_path}")
        self.get_logger().info(f"[CAM] model_rz_path  = {self.model_rz_path}")
        
        self.declare_parameter("class_names", ["ESP32", "L298N", "MB102"])

        # 기존 RZ_CENTERS: np.arange(-90+5, 70+5, 10) → [-85..75] (17개)
        self.declare_parameter("rz_centers", [-85, -75, -65, -55, -45, -35, -25, -15, -5,
                                             5, 15, 25, 35, 45, 55, 65, 75])

        self.declare_parameter("base_pick_coords", [-237.90, 20.0, 183.6, -174.98, 0.0, 0.0])

        # get_vision_rz ROI/조건(기존 유지)
        self.declare_parameter("rz_roi", [70, 330, 90, 390])  # y1,y2,x1,x2
        self.declare_parameter("rz_area_thresh", 500)

        # 앙상블 비율(기존: 0.8 vision + 0.2 ai)
        self.declare_parameter("w_vis", 0.8)
        self.declare_parameter("w_ai", 0.2)

        # 이미지 전송(기존 send_full_result와 동일 목적)
        self.declare_parameter("include_image", True)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("jpeg_size", 224)

        # busy 가드
        self._busy = False

        # -----------------------------
        # 카메라 오픈
        # -----------------------------
        self.cap = self._open_camera()

        # -----------------------------
        # 모델 로드(기존 로직 유지)
        # -----------------------------
        self.class_names = list(self.get_parameter("class_names").value)
        self.rz_centers = np.array(self.get_parameter("rz_centers").value, dtype=np.float32)

        self.device = None
        self.cls_m = None
        self.rz_m = None
        self.transform = None

        self._load_models()

        self.get_logger().info("✅ camera_vision_node ready")

    # ==================================================
    # camera
    # ==================================================
    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        idx = int(self.get_parameter("camera_index").value)
        idx = 0
        # cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
        if not cap.isOpened():
            self.get_logger().error("❌ VideoCapture open failed")
            return None
        return cap

    def _capture(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None

        flush = int(self.get_parameter("flush_frames").value)
        for _ in range(max(0, flush)):
            self.cap.read()
        return self.cap.read()
    

    def _resolve_model_path(self, p: str) -> str:
        """
        - p가 절대경로면 그대로 사용
        - 상대경로면: <share/ARM_PKG>/data/vision/<p> 로 해석
        """
        if not p:
            return p
        if os.path.isabs(p):
            return p

        share_dir = get_package_share_directory("ARM_PKG")
        return os.path.join(share_dir, "data", "vision", p)

    # ==================================================
    # models (기존 load_all_models 느낌 유지)
    # ==================================================

    def _load_models(self):
        enable_ai = bool(self.get_parameter("enable_ai").value)
        if not enable_ai:
            self.get_logger().warn("[CAM] enable_ai=False → AI 없이 동작(vision rz만 가능)")
            return
        if not TORCH_OK:
            self.get_logger().error("[CAM] torch/torchvision import 실패 → AI 비활성화")
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ✅ 여기 중요: resolve된 경로를 사용
            cls_path = getattr(self, "model_cls_path", None)
            rz_path  = getattr(self, "model_rz_path", None)

            self.get_logger().info(f"[CAM][DEBUG] cls_path={cls_path}")
            self.get_logger().info(f"[CAM][DEBUG] rz_path ={rz_path}")
            self.get_logger().info(f"[CAM][DEBUG] exists(cls)={os.path.exists(cls_path)}")
            self.get_logger().info(f"[CAM][DEBUG] exists(rz) ={os.path.exists(rz_path)}")

            # ✅ open 테스트 (여기서도 실패하면 진짜 파일/권한/경로 문제)
            with open(cls_path, "rb") as f:
                _ = f.read(16)
            with open(rz_path, "rb") as f:
                _ = f.read(16)

            # 분류 모델
            cls_m = models.resnet50(weights=None)
            cls_m.fc = nn.Linear(cls_m.fc.in_features, len(self.class_names))
            cls_m.load_state_dict(torch.load(cls_path, map_location=self.device))

            # Rz 모델
            rz_m = ResNetMultiTask(num_classes=len(self.rz_centers))
            rz_m.load_state_dict(torch.load(rz_path, map_location=self.device))

            for m in (cls_m, rz_m):
                m.to(self.device).eval()

            self.cls_m = cls_m
            self.rz_m = rz_m

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])

            self.get_logger().info(f"✅ AI 모델 로드 완료 (device={self.device})")

        except Exception as e:
            # ✅ filename이 있으면 같이 출력
            fn = getattr(e, "filename", None)
            self.get_logger().error(f"❌ 모델 로드 실패: {repr(e)} filename={fn}")
            self.cls_m = None
            self.rz_m = None
            self.transform = None


    # ==================================================
    # 기존 get_vision_rz (기능 유지)
    # ==================================================
    def _get_vision_rz(self, frame: np.ndarray) -> Tuple[Optional[float], float]:
        y1, y2, x1, x2 = self.get_parameter("rz_roi").value
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return None, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))

        rect = cv2.minAreaRect(c)
        (_, _), (w, h), angle = rect
        final_rz = (-angle + 90.0) if (w < h) else (-angle)
        final_rz = float(np.clip(final_rz, -90.0, 90.0))
        return final_rz, area

    # ==================================================
    # AI inference (기존 execute_mission의 1~4 단계 대응)
    # ==================================================
    def _infer(self, frame: np.ndarray) -> Tuple[str, float, Optional[float]]:
        """
        return: (module_type, confidence, ai_rz)
        """
        if self.cls_m is None or self.rz_m is None or self.transform is None:
            return "UNKNOWN", 0.0, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_t = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_out = self.cls_m(input_t)
            prob = torch.softmax(cls_out, 1)
            conf, idx = torch.max(prob, 1)

            # 기존: _, res_out = self.rz_m(input_t)
            _, res_out = self.rz_m(input_t)

            idx_i = int(idx.item())
            conf_f = float(conf.item())

            # ⚠️ 기존 담당자 코드 그대로: RZ_CENTERS[idx] 사용
            # (논리적으로는 rz_cls를 써야 맞을 수 있지만, 지금은 "기존 로직 유지"가 목표)
            center = float(self.rz_centers[idx_i]) if idx_i < len(self.rz_centers) else 0.0
            ai_rz = float(np.clip(center + float(res_out.item()), -90.0, 90.0))

            module_type = self.class_names[idx_i] if idx_i < len(self.class_names) else "UNKNOWN"

        return module_type, conf_f, ai_rz

    # ==================================================
    # 이미지 base64 (기존 send_full_result 용도)
    # ==================================================
    def _to_b64(self, frame: np.ndarray) -> str:
        if not bool(self.get_parameter("include_image").value):
            return ""

        size = int(self.get_parameter("jpeg_size").value)
        quality = int(self.get_parameter("jpeg_quality").value)
        
        size = 224
        quality = 80

        try:
            resized = cv2.resize(frame, (size, size))
            ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return ""
            return base64.b64encode(buffer).decode("utf-8")
        except Exception:
            return ""

    # ==================================================
    # ROS callback
    # ==================================================
    def _on_cmd(self, msg: String):
        raw = msg.data
        data = self._safe_json(raw)

        action = str(data.get("action", "")).strip()
        if action != "inspect_pick_zone":
            # 현재는 이 액션만 처리 (요구사항 범위)
            self.get_logger().warn(f"[CAM] ignore action={action} raw={raw}")
            return

        if self._busy:
            # 안전(중복 실행 방지)만 유지
            self.get_logger().warn("[CAM] busy → fail publish")
            self._publish_coord_fail("busy")
            self._publish_ai_fail("busy")
            return

        self._busy = True
        t0 = time.time()

        try:
            ret, frame = self._capture()
            if not ret or frame is None:
                self.get_logger().error("[CAM] capture_failed")
                self._publish_coord_fail("capture_failed")
                self._publish_ai_fail("capture_failed")
                return

            # 1) AI inference
            module_type, confidence, ai_rz = self._infer(frame)

            # 2) Vision rz
            vis_rz, area = self._get_vision_rz(frame)

            # 3) final_rz (기존 로직 유지)
            area_thresh = float(self.get_parameter("rz_area_thresh").value)

            if ai_rz is None and vis_rz is None:
                final_rz = None
            elif ai_rz is None:
                final_rz = float(vis_rz)
            elif vis_rz is None:
                final_rz = float(ai_rz)
            else:
                if (vis_rz is not None) and (area > area_thresh):
                    wv = float(self.get_parameter("w_vis").value)
                    wa = float(self.get_parameter("w_ai").value)
                    final_rz = float(np.clip(wv * vis_rz + wa * ai_rz, -90.0, 90.0))
                else:
                    final_rz = float(ai_rz)

            if final_rz is None:
                self.get_logger().error("[CAM] final_rz_none")
                self._publish_coord_fail("final_rz_none")
                self._publish_ai_fail("final_rz_none")
                return

            # 4) pick_coord 생성
            base_pick = list(self.get_parameter("base_pick_coords").value)
            if len(base_pick) < 6:
                self.get_logger().error("[CAM] invalid_base_pick_coords")
                self._publish_coord_fail("invalid_base_pick_coords")
                self._publish_ai_fail("invalid_base_pick_coords")
                return

            pick_coord = [float(x) for x in base_pick[:6]]
            pick_coord[5] = float(final_rz)

            # 5) publish (좌표 → go_move_node)
            self._publish_coord_success(pick_coord, final_rz)

            # 6) publish (AI 로그 → write_opcua_node)
            img_b64 = self._to_b64(frame)
            elapsed_ms = int((time.time() - t0) * 1000)
            self._publish_ai_success(
                module_type=module_type,
                confidence=confidence,
                pick_coord=pick_coord,
                final_rz=final_rz,
                img_b64=img_b64,
                elapsed_ms=elapsed_ms,
            )

        finally:
            self._busy = False

    # ==================================================
    # publish: coord (to go_move_node)
    # ==================================================
    def _publish_coord_success(self, pick_coord: List[float], final_rz: float):
        payload = {
            "status": "success",
            "pick_coord": pick_coord,
            "final_rz": float(final_rz),
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.coord_pub.publish(out)
        self.get_logger().info(f"[CAM] → /arm/camera_result: {out.data}")

    def _publish_coord_fail(self, reason: str):
        payload = {"status": "fail", "reason": reason}
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.coord_pub.publish(out)
        self.get_logger().info(f"[CAM] → /arm/camera_result: {out.data}")

    # ==================================================
    # publish: AI log (to write_opcua_node)
    # ==================================================
    def _publish_ai_success(
        self,
        module_type: str,
        confidence: float,
        pick_coord: List[float],
        final_rz: float,
        img_b64: str,
        elapsed_ms: int,
    ):
        payload = {
            "mode": "JOINT_DETECTION",
            "module_type": module_type,
            "classification_confidence": float(confidence),
            "pick_coord": [f"{c:.2f}" for c in pick_coord],
            "img": img_b64,
            "status": "arm_mission_success",
            "elapsed_ms": elapsed_ms,
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.ai_pub.publish(out)
        # self.get_logger().info(f"[CAM] → /arm/ai_result: {out.data}")

    def _publish_ai_fail(self, reason: str):
        payload = {
            "mode": "JOINT_DETECTION",
            "module_type": "UNKNOWN",
            "classification_confidence": 0.0,
            "pick_coord": [],
            "img": "",
            "status": "arm_mission_failure",
            "reason": reason,
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.ai_pub.publish(out)
        self.get_logger().info(f"[CAM] → /arm/ai_result: {out.data}")

    # ==================================================
    # utils / shutdown
    # ==================================================
    def _safe_json(self, raw: str) -> Dict[str, Any]:
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {"value": v}
        except Exception:
            return {"value": raw}

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmCameraVisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("camera_vision_node 종료(Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
