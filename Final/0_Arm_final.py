import torch
import cv2
import time
import os
import sys
import numpy as np
import json
import asyncio
import base64
import logging
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from asyncua import ua
from asyncua.client import Client as AsyncuaClient
from pymycobot import MyCobot320
from db_manager import DBManager

EXECUTE_MISSION_COUNT = 0
LOAD_OBJECT_COUNT = 2

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotArmMain")

# --- AI 모델 및 비전 설정 ---
CLASS_NAMES = ["ESP32", "L298N", "MB102"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CLS_PATH = "best_trck_obj_cls_model.pth"
MODEL_RZ_PATH = "best_trck_coords_tracking_model.pth"
RZ_CENTERS = np.arange(-90 + 5, 70 + 5 + 1e-6, 10, dtype=np.float32)

# --- 하드웨어 제어 파라미터 ---
PORT, BAUD = "COM3", 115200
MOVEMENT_SPEED = 70
PICK_Z_HEIGHT = 260
GRIPPER_SPEED = 50
GRIPPER_OPEN, GRIPPER_CLOSE = 65, 25
GRIPPER_DELAY = 1.0

# --- 비전 좌표 보정 (Pixel to MM) ---
CAMERA_INDEX = 0
TARGET_CENTER_U, TARGET_CENTER_V = 320, 180
PIXEL_TO_MM_X, PIXEL_TO_MM_Y = 0.526, -0.698

# --- 로봇 주요 포즈 (Angles & Coords) ---
CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90]
ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90]
INTERMEDIATE_POSE = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
BASE_PICK_COORDS = [-240, 20, 180, -174.98, 0, 0]
GLOBAL_TARGET_COORDS = [-114, -195, 250, 177.71, 0.22, 0]
GLOBAL_TARGET_TMP_COORDS = [-150.0, -224.4, 318.1, 176.26, 3.2, 3.02]

# --- OPC UA 설정 ---
OPCUA_SERVER_URL = "opc.tcp://172.30.1.61:4840/freeopcua/server/"
READ_METHOD_NODE = "ns=2;s=read_arm_go_move"

WRITE_OBJ_NODE = "ns=2;i=3"
WRITE_METHOD_NODE = "ns=2;s=write_send_arm_json"

WRITE_SINGLE_OBJ_NODE = "ns=2;i=3"
WRITE_SINGLE_METHOD_NODE = "ns=2;s=write_arm_place_single"

WRITE_COMPLETE_OBJ_NODE = "ns=2;i=3"
WRITE_COMPLETE_METHOD_NODE = "ns=2;s=write_arm_place_completed"

LOWER_RED_HSV1 = np.array([0, 100, 100])
UPPER_RED_HSV1 = np.array([15, 255, 255])
LOWER_RED_HSV2 = np.array([155, 100, 100])
UPPER_RED_HSV2 = np.array([179, 255, 255])
#

def round_coords(coords, precision=2):
    # 입력이 리스트나 튜플인 경우
    if isinstance(coords, (list, tuple, np.ndarray)):
        return [round(float(c), precision) for c in coords]
    # 입력이 단일 숫자인 경우
    return round(float(coords), precision)

class ResNetMultiTask(nn.Module):
    """Rz 추론을 위한 Multi-Task ResNet50 모델 구조"""
    def __init__(self, num_classes=17):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        common_fc = lambda out: nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, out)
        )
        self.cls_head = common_fc(num_classes)
        self.res_head = common_fc(1)

    def forward(self, x):
        x = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.cls_head(x), self.res_head(x)

def load_all_models():
    """분류 및 Rz 모델을 각각 로드하여 반환"""
    try:
        # 분류 모델 (3 Classes)
        cls_m = models.resnet50(weights=None)
        cls_m.fc = nn.Linear(cls_m.fc.in_features, 3)
        cls_m.load_state_dict(torch.load(MODEL_CLS_PATH, map_location=DEVICE))
        
        # Rz 추론 모델 (17 Classes)
        rz_m = ResNetMultiTask(num_classes=17)
        rz_m.load_state_dict(torch.load(MODEL_RZ_PATH, map_location=DEVICE))
        
        for m in [cls_m, rz_m]: m.to(DEVICE).eval()
        logger.info("✅ 모든 AI 모델 로드 완료")
        return cls_m, rz_m
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        return None, None

# 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 

def get_vision_rz(frame):
    roi = frame[70:330, 90:390]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 흰색 배경 위의 물체를 잡기 위한 마스크 (필요시 범위 조정)
    mask = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        logger.warning(f"테두리 미검출... 재검출 시도 중")
        return 0
    
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect

    # --- 디버깅용 시각화 코드 시작 ---
    # box = cv2.boxPoints(rect)
    # box = np.int32(box)
    # debug_img = roi.copy()
    # cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2) # 초록색 박스
    
    # # 각도 계산 로직 디버깅 출력
    # # if w < h:
    # #     final_rz = -angle + 90
    # # else:
    # #     final_rz = -angle
    
    # # 화면에 정보 표시
    # cv2.putText(debug_img, f"W:{w:.1f} H:{h:.1f} Ang:{angle:.1f}", (10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    # cv2.putText(debug_img, f"Final Rz(=angle): {angle:.1f}", (10, 60), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # cv2.imshow("Vision Debug (ROI)", debug_img) # ROI 내부 확인
    # cv2.waitKey(1) 
    # --- 디버깅용 시각화 코드 끝 ---
    return round_coords(angle, 2)
    # return np.clip(final_rz, -90, 90), (cx + 90, cy + 70), cv2.contourArea(cnt)

async def send_img_result(module_type, confidence, pick_coord, status, image=None):
    """결과 데이터를 JSON으로 변환하여 OPC UA로 전송"""
    img_b64 = ""
    if image is not None and status != "arm_mission_failure":
        _, buffer = cv2.imencode('.jpg', cv2.resize(image, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "module_type": module_type,
        "classification_confidence": confidence,
        "pick_coord": [f"{c:.2f}" for c in pick_coord],
        "img": img_b64,
        "status": status
    }

    try:
        async with AsyncuaClient(OPCUA_SERVER_URL) as client:
            obj = client.get_node(WRITE_OBJ_NODE)
            method = client.get_node(WRITE_METHOD_NODE)
            await obj.call_method(method.nodeid, ua.Variant(json.dumps(payload), ua.VariantType.String))
            logger.info(f"📡 OPC UA 결과 송신: {status}")
    except Exception as e:
        logger.error(f"📡 송신 오류: {e}")

async def send_single_result():
    payload = {
        "status": "arm_place_single"
    }

    try:
        async with AsyncuaClient(OPCUA_SERVER_URL) as client:
            obj = client.get_node(WRITE_SINGLE_OBJ_NODE)
            method = client.get_node(WRITE_SINGLE_METHOD_NODE)
            await obj.call_method(method.nodeid, ua.Variant(json.dumps(payload), ua.VariantType.String))
            logger.info(f"📡 OPC UA 결과 송신: {payload}")
    except Exception as e:
        logger.error(f"📡 송신 오류: {e}")

async def send_completed_result():
    payload = {
        "status": "arm_place_completed"
    }

    try:
        async with AsyncuaClient(OPCUA_SERVER_URL) as client:
            obj = client.get_node(WRITE_COMPLETE_OBJ_NODE)
            method = client.get_node(WRITE_COMPLETE_METHOD_NODE)
            await obj.call_method(method.nodeid, ua.Variant(json.dumps(payload), ua.VariantType.String))
            logger.info(f"📡 OPC UA 결과 송신: {payload}")
    except Exception as e:
        logger.error(f"📡 송신 오류: {e}")

#

def find_red_center(frame):
    """ 주어진 이미지 프레임에서 가장 큰 빨간색 영역의 중심 픽셀 (u, v)를 찾고 윤곽선을 반환합니다. """
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 두 개의 빨간색 범위 마스크를 합치기 (0~10도, 160~179도)
    mask1 = cv2.inRange(hsv_frame, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask2 = cv2.inRange(hsv_frame, LOWER_RED_HSV2, UPPER_RED_HSV2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 윤곽선 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 50: # 최소 면적 필터링
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                return (center_x, center_y, largest_contour)
                
    return (None, None, None) # 검출 실패 시 None 반환

def convert_pixel_to_robot_move(current_center_u, current_center_v):
    global TARGET_CENTER_U, TARGET_CENTER_V, PIXEL_TO_MM_X, PIXEL_TO_MM_Y
    
    delta_u_pixel = current_center_u - TARGET_CENTER_U
    delta_v_pixel = current_center_v - TARGET_CENTER_V
    
    delta_X_mm = delta_u_pixel * PIXEL_TO_MM_X
    delta_Y_mm = delta_v_pixel * PIXEL_TO_MM_Y
    
    final_delta_X = -delta_X_mm
    final_delta_Y = -delta_Y_mm
    
    return final_delta_X, final_delta_Y, delta_u_pixel, delta_v_pixel

#

class SubHandler:
    def __init__(self, mc, cls_m, rz_m):
        self.mc = mc
        self.cls_m, self.rz_m = cls_m, rz_m
        self.db = DBManager() # DB 매니저 초기화
        self.current_mission_id = None
    
    async def wait_until_pose(self, target_angles, tolerance=1.0):
        """실제 각도가 목표 각도에 근접할 때까지 대기"""
        while True:
            curr_angles = self.mc.get_angles()
            if curr_angles:
                # 모든 관절의 오차 합 계산 (L1 Norm)
                error = sum(abs(c - t) for c, t in zip(curr_angles, target_angles))
                if error < tolerance:
                    break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5) # 물리적 진동 감쇄를 위한 추가 여유 시간

    def datachange_notification(self, node, val, data):
        asyncio.create_task(self.process_command(val))

    async def wait_stop(self, delay=0.5):
        while await asyncio.to_thread(self.mc.is_moving):
            await asyncio.sleep(0.2)
        await asyncio.sleep(delay)

    async def process_command(self, val):
        try:
            try:
                cmd_data = json.loads(val)
                cmd = cmd_data.get("move_command")
            except:
                cmd = str(val)

            # 2. 무의미한 호출 필터링 (명령이 있을 때만 DB 시작)
            if cmd not in ["go_home", "mission_start"]:
                return
            
            if self.current_mission_id is None:
                self.current_mission_id = await self.db.insert_mission_start()
                logger.info(f"🆕 미션 ID 자동 생성 (ID: {self.current_mission_id})")

            logger.info(f"📥 수신 명령: {cmd} (Mission ID: {self.current_mission_id})")

            # 4. 명령 실행
            if cmd == "go_home":
                await self.move_home()
            elif cmd == "mission_start":
                if EXECUTE_MISSION_COUNT > 0 and EXECUTE_MISSION_COUNT % LOAD_OBJECT_COUNT == 0:
                    self.current_mission_id = await self.db.insert_mission_start()
                    logger.info(f"🆕 새로운 미션 세션 시작 (ID: {self.current_mission_id})")
                
                await self.execute_mission()

        except Exception as e:
            logger.error(f"❌ 명령 처리 중 오류: {e}")
            # [수정] current_mission_id가 확실히 있을 때만 로그 시도
            if self.current_mission_id is not None:
                try:
                    await self.db.insert_arm_log(self.current_mission_id, 'ERROR', result_status='FAIL', result_message=str(e))
                    await self.db.update_mission_status(self.current_mission_id, 'ERROR')
                except:
                    logger.error("DB 로그 기록마저 실패했습니다.")
                    
    async def move_home(self):
        self.mc.sync_send_coords(INTERMEDIATE_POSE, MOVEMENT_SPEED)
        await self.db.insert_arm_log(self.current_mission_id, 'MOVE', target_pose=INTERMEDIATE_POSE, result_status='SUCCESS', description="임시 Conveyor 캡처 포즈로 이동")
        # await self.wait_stop()
        self.mc.sync_send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
        await self.db.insert_arm_log(self.current_mission_id, 'HOME', target_pose=CONVEYOR_CAPTURE_POSE, result_status='SUCCESS', description="Conveyor 캡처 포즈로 이동")
        # await self.wait_stop()

    async def execute_mission(self):
        global EXECUTE_MISSION_COUNT
        EXECUTE_MISSION_COUNT += 1

        # await self.wait_stop()

        logger.info("📸 촬영 위치로 이동 중...")
        self.mc.sync_send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)

        await self.wait_until_pose(CONVEYOR_CAPTURE_POSE)
        # await self.wait_stop()
        
        # for _ in range(10):
            # cap.grab()
        
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        # 1. Capture & AI Inference
        ret, send_frame = cap.read()
        print("\n\n이미지 찍었음\n\n")
        
        if not ret: return
        
        # AI & Vision Ensemble (Pick Angle)
        input_t = transform(Image.fromarray(cv2.cvtColor(send_frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cls_out = self.cls_m(input_t)
            conf, idx = torch.max(torch.softmax(cls_out, 1), 1)
            # _, res_out = self.rz_m(input_t)
            # ai_rz = np.clip(RZ_CENTERS[idx.item()] + res_out.item(), -90, 90)
        
        # final_rz, _, area = get_vision_rz(frame)
        final_rz = 90-get_vision_rz(send_frame)
        # final_rz = final_rz - 90
        # final_rz = (0.8 * vis_rz + 0.2 * ai_rz) if vis_rz is not None and area > 500 else ai_rz
        
        # 2. Pick Action
        pick_pose = list(BASE_PICK_COORDS)
        pick_pose[5] = final_rz
        logger.info(f"final_rz: {final_rz}")

        # cv2.imshow("send_frame", send_frame)
        await send_img_result(
            module_type=CLASS_NAMES[idx.item()], 
            confidence=conf.item(), 
            pick_coord=pick_pose, 
            status="이미지 전송 완료 -> 동작 시작", 
            image=send_frame)
        print("\n\nOPCUA 송신 완료\n\n")
        # 동작 시퀀스 (Safety -> Pick -> Close)
        for z_off in [50, 0]:
            p = list(pick_pose)
            p[2] += z_off
            self.mc.sync_send_coords(p, MOVEMENT_SPEED - 20)
            # await self.wait_stop()
        await self.db.insert_arm_log(self.current_mission_id, 'MOVE', target_pose=BASE_PICK_COORDS, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="Pick 포즈로 이동")
        print("\n\n움직임\n\n")
        self.mc.set_gripper_value(GRIPPER_CLOSE, GRIPPER_SPEED)
        await asyncio.sleep(GRIPPER_DELAY)
        await self.db.insert_arm_log(self.current_mission_id, 'GRIPPER_CLOSE', target_pose=GRIPPER_CLOSE, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="그리퍼 닫기 완료")
        
        await self.db.insert_arm_log(self.current_mission_id, 'PICK', target_pose=pick_pose, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="Pick 완료")
        cap.release()
#__________________________End pick process__________________________

        # 3. Place Action (Vision-Guided)
        self.mc.sync_send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
        await self.db.insert_arm_log(self.current_mission_id, 'MOVE', target_pose=ROBOTARM_CAPTURE_POSE, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="로봇 암 캡처 포즈로 이동")
        # await self.wait_stop()
        
        # await asyncio.sleep(1.5)
        # 카메라 잔상 제거를 위한 버퍼 비우기
        # for _ in range(10):
        #     cap.grab()
        
        max_retries = 20  # 최대 20번 시도 (약 10~20초)
        retry_count = 0
        center_u, center_v = None, None
        # 현재 프레임 캡처 및 빨간색 중심점 찾기
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        while retry_count < max_retries:
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ Place용 프레임 수신 실패, 재시도 중...")
                await asyncio.sleep(0.5)
                retry_count += 1
                continue

            # 빨간색 영역 검출 (기존 find_red_center 함수 호출)
            center_u, center_v, _ = find_red_center(frame)
            
            if center_u is not None:
                # cv2.imshow("frame", frame)
                logger.info(f"🎯 빨간색 물체 검출 성공! (u: {center_u}, v: {center_v})")
                break
                
            retry_count += 1
            logger.warning(f"🔴 빨간색 물체 미검출... 재검출 시도 중 ({retry_count}/{max_retries})")
            await asyncio.sleep(0.5)  # 0.5초 대기 후 다시 확인

        if center_u is None:
            logger.error(f"⚠️ {max_retries}회 시도 후에도 물체를 찾지 못했습니다. 미션을 중단합니다.")
            cap.release()
            # 안전을 위해 홈으로 복귀하거나 에러 처리
            return

        # 픽셀 오차 -> 로봇 이동량(mm) 변환
        delta_X_mm, delta_Y_mm, _, _ = convert_pixel_to_robot_move(center_u, center_v)
        
        # 최종 목표 좌표 생성 (기준 좌표 + 보정값)
        final_place_coords = list(GLOBAL_TARGET_COORDS)
        final_place_coords[0] += delta_X_mm
        final_place_coords[1] += delta_Y_mm
        final_place_coords[2] = PICK_Z_HEIGHT  # 내려놓을 높이

        final_place_coords = [round(x, 2) for x in final_place_coords]

        final_place_coords[1] += 20
        logger.info(f"✅ Place 목표 확정: X:{final_place_coords[0]:.2f}, Y:{final_place_coords[1]:.2f}")

        # ---------------------------------------------------------
        # 6. Place 동작 실행 (이동 및 그리퍼 제어)
        # ---------------------------------------------------------
        
        # 안전 이동을 위한 임시 포즈 (Z축 높은 곳)
        safe_place_tmp = list(GLOBAL_TARGET_TMP_COORDS)

        # [STEP 1] Place 구역 위 안전 포즈로 이동
        logger.info(f"⬆️ Place 안전 포즈로 이동 중...{safe_place_tmp}")
        self.mc.sync_send_coords(safe_place_tmp, MOVEMENT_SPEED - 20)
        await self.db.insert_arm_log(self.current_mission_id, 'MOVE', target_pose=safe_place_tmp, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="[Place] 안전 포즈로 이동")
        # await self.wait_stop()

        # [STEP 2] 계산된 정밀 좌표로 하강
        logger.info(f"⬇️ 정밀 Place 지점으로 하강 중...{final_place_coords}")
        self.mc.sync_send_coords(final_place_coords, MOVEMENT_SPEED - 30)
        await self.db.insert_arm_log(self.current_mission_id, 'MOVE', target_pose=final_place_coords, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="Place 작업 시작")
        # await self.wait_stop()

        # [STEP 3] 그리퍼 열기 (내려놓기)
        logger.info("✊ 그리퍼 개방 (Place 완료)")
        self.mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED)
        await self.db.insert_arm_log(self.current_mission_id, 'GRIPPER_OPEN', target_pose=GRIPPER_OPEN, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="그리퍼 열기 완료")
        # await self.wait_stop()

        await self.db.insert_arm_log(self.current_mission_id, 'PLACE', target_pose=final_place_coords, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="Place 완료")

        # [STEP 4] 충돌 방지를 위해 다시 위로 복귀
        logger.info("⬆️ 복귀: 다시 안전 포즈로 이동")
        self.mc.sync_send_coords(safe_place_tmp, MOVEMENT_SPEED)
        await self.db.insert_arm_log(self.current_mission_id, 'MOVE', target_pose=safe_place_tmp, result_status='SUCCESS', module_type=CLASS_NAMES[idx.item()], description="[Place] 완료 안전 포즈로 이동")
        # await self.wait_stop()
        
        logger.info("🏁 모든 미션이 성공적으로 완료되었습니다.")

        if EXECUTE_MISSION_COUNT % LOAD_OBJECT_COUNT == 0:
            await send_completed_result()
            logger.info("📡 OPC UA 전송: send_completed_result")
            await self.db.update_mission_status(self.current_mission_id, 'DONE')
            logger.info(f"✅ 미션 완료 기록 (ID: {self.current_mission_id})")
        else:
            await send_single_result()
            logger.info("📡 OPC UA 전송: send_single_result")
# 

async def main():
    cls_m, rz_m = load_all_models()
    if not cls_m: return

    try:
        mc = MyCobot320(PORT, BAUD)
        mc.power_on()
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 초기 상태: 파워 ON (서보 잠금)")

        # 그리퍼 초기화 로직
        mc.set_gripper_mode(0)
        mc.init_electric_gripper()
        time.sleep(0.5)
        mc.set_electric_gripper(0)
        mc.set_gripper_value(GRIPPER_OPEN, GRIPPER_SPEED, 1) # GRIPPER_OPEN_VALUE (85)로 열림
        time.sleep(0.5)
        print(f"✅ 그리퍼 초기화 완료. 위치: **{GRIPPER_OPEN} (열림)**.")
        
        async with AsyncuaClient(OPCUA_SERVER_URL) as client:
            handler = SubHandler(mc, cls_m, rz_m)
            sub = await client.create_subscription(100, handler)
            await sub.subscribe_data_change(client.get_node(READ_METHOD_NODE))
            
            logger.info("🚀 시스템 가동 중... 명령 대기")
            while True: await asyncio.sleep(0.5)
            
    except Exception as e:
        logger.error(f"시스템 오류: {e}")
    finally:
        if 'mc' in locals(): mc.close()

if __name__ == "__main__":
    asyncio.run(main())