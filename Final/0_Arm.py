import torch
import cv2
import time
import os
import sys
import numpy as np
import json
import asyncio
from asyncua import ua
from asyncua.client import Client as AsyncuaClient
from pymycobot import MyCobot320
from torchvision import transforms
from PIL import Image
import base64
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# 📌 AI 모델 및 상수 설정 (기존과 동일)
# ===============================================
CLASS_NAMES = ["ESP32", "L298N(Motor)", "MB102(Power)"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu") # 비동기 환경을 고려해 CPU로 설정
MOBILENET_MEAN = [0.485, 0.456, 0.406]
MOBILENET_STD = [0.229, 0.224, 0.225]
MODEL_WEIGHTS_PATH = "checkpoint_mobilenetv3_classifier_e5_acc1.0000.pth"

# ⚙️ MyCobot 및 비전 시스템 설정 (기존과 동일)
PORT = "COM3"
BAUD = 115200

MOVEMENT_SPEED = 70
GRIPPER_SPEED = 50
SEQUENTIAL_MOVE_DELAY = 3
GRIPPER_ACTION_DELAY = 1

CAMERA_INDEX = 0
roi_start = (0, 0)
roi_end = (640, 360)
TARGET_CENTER_U = 320
TARGET_CENTER_V = 180

PIXEL_TO_MM_X = 0.526
PIXEL_TO_MM_Y = -0.698

MAX_PIXEL_ERROR = 5
PICK_Z_HEIGHT = 260

GRIPPER_OPEN_VALUE = 85
GRIPPER_CLOSED_VALUE = 25

LOWER_HSV = np.array([0, 0, 0])
UPPER_HSV = np.array([179, 255, 190])

CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90]
ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90]

INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

TEST_PICK_POSE_WIDTH = [-237.90, 20, 183.6, -174.98, 0, 0]
TEST_PICK_POSE_HEIGHT = [-237.90, 20, 183.6, -174.98, 0, 90]

# --- Place 변수
LOWER_RED_HSV1 = np.array([0, 100, 100])
UPPER_RED_HSV1 = np.array([15, 255, 255])
LOWER_RED_HSV2 = np.array([155, 100, 100])
UPPER_RED_HSV2 = np.array([179, 255, 255])

GLOBAL_TARGET_COORDS = [-114, -195, 250, 177.71, 0.22, 0]
GLOBAL_TARGET_TMP_COORDS = [-150.0, -224.4, 318.1, 176.26, 3.2, 3.02]

# --- 🎯 OPC UA 수신/송신 설정 (수정된 부분) ---
OPCUA_READ_URL = "opc.tcp://172.30.1.61:0630/freeopcua/server/"
OPCUA_WRITE_URL = "opc.tcp://172.30.1.61:0630/freeopcua/server/"

# 📌 읽기(구독) 노드 ID
READ_OBJECT_NODE_ID = "ns=2;i=3"
READ_METHOD_NODE_ID = "ns=2;s=read_arm_go_move"

# 📌 쓰기(Method Call) 노드 ID
WRITE_OBJECT_NODE_ID = "ns=2;i=3"
WRITE_METHOD_NODE_ID = "ns=2;s=write_send_arm_json"

# 전역 객체 (메인 및 핸들러에서 사용)
mc = None
cap = None
ai_model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MOBILENET_MEAN, std=MOBILENET_STD)
])

# ===============================================
# 🧠 AI 모델 함수 (기존과 동일)
# ===============================================

def load_model(model_path, num_classes):
    """학습된 모델 가중치를 로드하고 평가 모드로 설정합니다."""
    try:
        # models.mobilenet_v3_small 함수를 재사용
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', weights=None)
        
        # 최종 분류층(Classifier) 재정의
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        
        # 저장된 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # 평가 모드 설정
        print(f"✅ AI 모델 로드 완료: {model_path} ({DEVICE})")
        return model
    except FileNotFoundError:
        logger.error(f"\n❌ 오류: 모델 가중치 파일({model_path})을 찾을 수 없습니다. 경로를 확인하세요.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 모델 로드 중 예기치 않은 오류 발생: {e}")
        sys.exit(1)

def classify_object(model, transform, cropped_img):
    """크롭된 이미지로 객체 분류 추론을 수행합니다."""
    if cropped_img is None or cropped_img.size == 0:
        return "Unknown", 0.0

    # OpenCV (BGR) -> RGB
    rgb_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    # NumPy 배열 -> PIL Image -> Tensor로 변환 및 정규화
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf_score, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence = conf_score.item()
        
    return predicted_class, confidence

# ===============================================
# 🛠️ 로봇 및 비전 제어 함수 (수정 없음)
# ===============================================

def convert_pixel_to_robot_move(current_center_u, current_center_v):
    global TARGET_CENTER_U, TARGET_CENTER_V, PIXEL_TO_MM_X, PIXEL_TO_MM_Y
    
    delta_u_pixel = current_center_u - TARGET_CENTER_U
    delta_v_pixel = current_center_v - TARGET_CENTER_V
    
    delta_X_mm = delta_u_pixel * PIXEL_TO_MM_X
    delta_Y_mm = delta_v_pixel * PIXEL_TO_MM_Y
    
    final_delta_X = -delta_X_mm
    final_delta_Y = -delta_Y_mm
    
    return final_delta_X, final_delta_Y, delta_u_pixel, delta_v_pixel

def find_object_center(frame):
    """ 물체 중심 좌표를 찾고, 물체 영역을 크롭한 이미지를 함께 반환합니다. """
    global LOWER_HSV, UPPER_HSV, roi_start, roi_end
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask_full = cv2.inRange(hsv_frame, LOWER_HSV, UPPER_HSV)
    
    roi_mask = np.zeros(color_mask_full.shape, dtype=np.uint8)
    roi_mask[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = 255
    
    color_mask = cv2.bitwise_and(color_mask_full, color_mask_full, mask=roi_mask)
    
    kernel = np.ones((5, 5), np.uint8) 
    color_mask = cv2.erode(color_mask, kernel, iterations=1)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    inverted_mask = cv2.bitwise_not(color_mask)
    final_mask = cv2.bitwise_and(inverted_mask, inverted_mask, mask=roi_mask)
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 1000:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                
                rect = cv2.minAreaRect(largest_contour)
                
                # 물체 영역의 Bounding Box 좌표 (AI 추론을 위한 크롭 영역)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 이미지 경계를 벗어나지 않도록 클램프
                x = max(0, x - 10) 
                y = max(0, y - 10)
                x_end = min(frame.shape[1], x + w + 20)
                y_end = min(frame.shape[0], y + h + 20)
                
                # 물체 영역 크롭
                cropped_object_img = frame[y:y_end, x:x_end]
                return (center_x, center_y, largest_contour, rect, cropped_object_img) # 크롭된 이미지 추가 반환
            
    return (None, None, None, None, None)

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

# ===============================================
# 🤖 로봇 비동기 헬퍼 함수 (안전 마진 추가)
# ===============================================

async def wait_until_stopped(mc, safety_delay=2.0):
    """ 로봇이 움직임을 완전히 멈추고 안전 마진 시간만큼 대기합니다. """
    logger.info("... 로봇 움직임 완료 대기 중 (is_moving 체크)...")

    # 1. is_moving이 False를 반환할 때까지 대기
    while await asyncio.to_thread(mc.is_moving):
        await asyncio.sleep(0.2)
        
    # 2. 움직임이 멈춘 후, 로봇 제어기가 다음 명령을 받을 준비가 될 시간을 확보 (안전 마진)
    logger.info(f"... 움직임 중지 확인. 안전 마진 {safety_delay}초 추가 대기...")
    await asyncio.sleep(safety_delay) 
    
    return True

async def place_coords_calculator(cap):
    """ 
    [Vision-Guided] 검출된 빨간색 구역의 중심을 기준으로 배치할 최종 목표 좌표를 계산하여 반환합니다.
    """
    print("☆★☆★☆★☆★ place_coords_calculator: Place 목표 좌표 계산 시작")
    global GLOBAL_TARGET_COORDS, MOVEMENT_SPEED, PICK_Z_HEIGHT
    
    # 1. 이미지 캡처 및 중심 찾기
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 수신 실패. 좌표 계산 중지.")
        return False, None
    
    DATA_DIR = "place_capture"
    filename = f"place_calc_frame.jpg"
    save_path = os.path.join(DATA_DIR, filename)
    cv2.imwrite(save_path, frame)

    print(f"🖼️ Place 계산 프레임 저장 완료: {save_path}")
    center_u, center_v, _ = find_red_center(frame)
    
    if center_u is None:
        print(f"🔴 빨간색 물체 미검출. 좌표 계산 중지.")
        return False, None

    # 2. 오차 계산 및 MM 변환
    delta_X_mm, delta_Y_mm, delta_u_pixel, delta_v_pixel = convert_pixel_to_robot_move(center_u, center_v)
    
    total_pixel_error = np.sqrt(delta_u_pixel**2 + delta_v_pixel**2)
    
    print(f"\n--- 🤖 Vision-Guided 정렬 계산 (Single Shot) ---")
    print(f"  [Detect] 픽셀 오차: {total_pixel_error:.2f}px (U: {delta_u_pixel}, V: {delta_v_pixel})")
    print(f"  [Move] 필요한 이동량: X:{delta_X_mm:.2f}mm, Y:{delta_Y_mm:.2f}mm")

    # 3. 최종 목표 좌표 계산
    final_place_coords = list(GLOBAL_TARGET_COORDS) # 기준 좌표 복사
    
    # 픽셀 오차를 MM으로 변환한 만큼 로봇 좌표에 추가하여 '정렬된' 목표 좌표를 생성
    final_place_coords[0] += delta_X_mm # X축 이동 명령 적용
    final_place_coords[1] += delta_Y_mm # Y축 이동 명령 적용
    
    # Z축 높이는 미리 설정된 픽업 높이로 고정
    final_place_coords[2] = PICK_Z_HEIGHT 

    print(f"✅ 목표 좌표 계산 완료. 최종 좌표: X:{final_place_coords[0]:.2f}, Y:{final_place_coords[1]:.2f}, Z:{PICK_Z_HEIGHT:.2f}")
    
    # 4. 계산된 좌표 반환
    return True, final_place_coords

# ===============================================
# 🧠 AI/Vision 정보 수집 함수 (로봇 동작 제거)
# ===============================================

async def pick_data_collector(cap, ai_model):
    """
    로봇의 Pick 동작에 필요한 비전/AI 정보를 수집하고 반환합니다. (로봇 동작 제거)
    """
    global TEST_PICK_POSE_WIDTH, TEST_PICK_POSE_HEIGHT, transform

    # 현재 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        logger.error("❌ 카메라 프레임 읽기 실패. 데이터 수집 중단.")
        return False, "Unknown", 0.0, [0.0]*6, None
    
    center_x, center_y, largest_contour, rect, cropped_img = find_object_center(frame)

    if rect is None:
        logger.error("❌ 물체를 찾을 수 없습니다. 데이터 수집 중단.")
        return False, "Unknown", 0.0, [0.0]*6, None
    
    # AI 객체 분류 수행
    predicted_class, confidence = classify_object(ai_model, transform, cropped_img)
    print(f"\n🧠 AI 분류 결과: **{predicted_class}** (신뢰도: {confidence*100:.2f}%)")

    (center_u, center_v), (w, h), angle = rect
    
    # 픽업 자세 결정 (가로/세로)
    if w > h:
        target_pose = list(TEST_PICK_POSE_WIDTH)
        logger.info(f"📐 물체 장축: 가로. 최종 Pose: 둘 다 TEST_PICK_POSE_WIDTH 선택.")
    else: 
        target_pose = list(TEST_PICK_POSE_WIDTH)
        logger.info(f"📐 물체 장축: 세로. 최종 Pose: 둘 다 TEST_PICK_POSE_WIDTH 선택.")
    
    # 사용자 지정 crop 후 server로 송신할 이미지
    send_img = frame[30:400, 30:340]
    
    return True, predicted_class, confidence, target_pose, send_img


# ===============================================
# 📡 OPC UA 통신 함수 (기존과 동일)
# ===============================================

async def send_full_result(module_type: str, confidence: float, pick_coord: list, status: str, image_to_send: np.ndarray = None):
    """
    분류, 픽업 결과 및 미션 상태를 JSON 형태로 묶어 OPC UA 서버에 한 번에 송신합니다.
    """
    global OPCUA_WRITE_URL, WRITE_OBJECT_NODE_ID, WRITE_METHOD_NODE_ID
    
    # --- 이미지 처리 및 인코딩 로직 (기존과 동일) ---
    base64_img_str = ""
    if image_to_send is not None and status != "arm_mission_failure": # 실패 시 이미지를 보내지 않아 트래픽 절약
        try:
            resized_img = cv2.resize(image_to_send, (224, 224), interpolation=cv2.INTER_AREA)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80] 
            _, buffer = cv2.imencode('.jpg', resized_img, encode_param)
            base64_img_bytes = base64.b64encode(buffer)
            base64_img_str = base64_img_bytes.decode('utf-8')
            logger.info(f"🖼️ 이미지 인코딩 완료. Base64 문자열 길이: {len(base64_img_str)}")
        except Exception as e:
            logger.error(f"이미지 인코딩 중 오류 발생: {e}")
            base64_img_str = ""
    
    # 📌 통합된 JSON 데이터 구성 (status 필드 추가)
    vision_result = {
        "module_type": module_type,
        "classification_confidence": confidence,
        "pick_coord": [f"{c:.2f}" for c in pick_coord],
        "pick_coord_confidence": 0.9984073221683503,
        "img": base64_img_str,
        "status": status # ⬅️ 미션 상태 통합
    }
    json_str = json.dumps(vision_result)

    # 🚀 클라이언트 전송 데이터 확인
    print("\n========================================================")
    print(f"🚀 [FULL RESULT] 클라이언트가 서버로 전송하는 최종 통합 JSON:")
    print(json_str)
    print("========================================================\n")
    
    try:
        async with AsyncuaClient(OPCUA_WRITE_URL) as client:
            obj = client.get_node(WRITE_OBJECT_NODE_ID)
            method_node = client.get_node(WRITE_METHOD_NODE_ID)
            json_variant = ua.Variant(json_str, ua.VariantType.String)

            print(f"[OPC UA WRITE - FULL_RESULT] call_method(Module: {module_type}, Status: {status}) (Method: {WRITE_METHOD_NODE_ID})")
            result_code, result_message = await obj.call_method(
                method_node.nodeid,
                json_variant
            )
            logger.info(f"OPC UA 통합 결과 송신 완료. ResultCode: {result_code}")

    except Exception as e:
        logger.error(f"OPC UA 통합 결과 송신 중 오류 발생: {e}")

def sync_flush_camera_buffer(cap, num_frames=10):
        for _ in range(num_frames):
            cap.read() # 프레임을 읽어 버림
            time.sleep(0.01)

class SubHandler:
    
    def __init__(self, mycobot_instance, camera_instance, ai_model_instance):
        self.mc = mycobot_instance
        self.cap = camera_instance
        self.ai_model = ai_model_instance
        logger.info("SubHandler 초기화 완료.")
        

    def datachange_notification(self, node, val, data):
        """데이터 변경 알림 시 호출되는 비동기적 콜백 함수"""
        asyncio.create_task(self.execute_command_and_respond(val))
    
    async def execute_command_and_respond(self, val):
        """
        [최종 수정] 명령을 파싱하고 MyCobot 동작을 모두 수행합니다.
        vision-guided place 이동 명령까지 여기서 처리합니다.
        """
        global SEQUENTIAL_MOVE_DELAY, MOVEMENT_SPEED, GRIPPER_OPEN_VALUE, GRIPPER_CLOSED_VALUE, GRIPPER_SPEED, GRIPPER_ACTION_DELAY

        print(f"\n■□■□■□■□■□■□[OPC UA READ] 수신 값: {val}")

        command = None
        if isinstance(val, str):
            try:
                json_data = json.loads(val)
                if "move_command" in json_data:
                    command = json_data["move_command"]
            except json.JSONDecodeError:
                command = val # Ready 같은 일반 문자열도 command로 간주

        if not command or self.mc is None or self.cap is None:
            logger.warning(f"-> 로봇/카메라 연결 문제 또는 알 수 없는 명령: {command}")
            return
        
        # 3. MyCobot 동작 수행 및 응답
        if command == "go_home":
            logger.info("-> MyCobot: go_home 명령 수행 (CONVEYOR_CAPTURE_POSE로 이동)")
            
            # 중간 포즈로 이동
            await asyncio.to_thread(self.mc.send_coords, INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            await wait_until_stopped(self.mc)
            
            # 컨베이어 캡처 포즈로 이동
            await asyncio.to_thread(self.mc.send_angles, CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            await wait_until_stopped(self.mc)
            logger.info("✅ go_home (CONVEYOR_CAPTURE_POSE) 이동 완료.")
        
        elif command == "mission_start":
            logger.info("-> MyCobot: mission_start 명령 수행 (Vision-Guided Pick 시작)")
            
            # 1. Vision/AI 데이터 수집 (Pick 목표 좌표)
            success, module_type, confidence, target_pick_pose, send_img = await pick_data_collector(self.cap, self.ai_model)

            if not success:
                logger.error("❌ 데이터 수집/물체 검출 실패. OPC UA 실패 보고 송신.")
                await send_full_result(
                    module_type=module_type, confidence=confidence, 
                    pick_coord=target_pick_pose, status="arm_mission_failure"
                )
                return 

            # 2. OPC UA 결과 전송 (로봇 동작 직전에 정보 전송)
            await send_full_result(
                module_type=module_type, confidence=confidence, 
                pick_coord=target_pick_pose, status="arm_mission_success", 
                image_to_send=send_img
            )

            # 3. 픽업 동작 시퀀스 시작
            safe_pick_pose = list(target_pick_pose)
            safe_pick_pose[2] += 50 
            
            logger.info(f"⬆️ 안전 포즈로 이동: Z:{safe_pick_pose[2]:.2f}")
            await asyncio.to_thread(self.mc.send_coords, safe_pick_pose, MOVEMENT_SPEED)
            await wait_until_stopped(self.mc)

            logger.info(f"\n⬇️ 픽업 시작: Z:{target_pick_pose[2]:.2f} 하강.")
            await asyncio.to_thread(self.mc.send_coords, target_pick_pose, MOVEMENT_SPEED - 30)
            await wait_until_stopped(self.mc)
            
            await asyncio.to_thread(self.mc.set_gripper_value, GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
            await asyncio.sleep(GRIPPER_ACTION_DELAY)
            logger.info("✅ 그리퍼 닫기 완료 (Pick).")

            # 4. 중간 포즈 이동 (Place 준비)
            await asyncio.to_thread(self.mc.send_angles, CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            await wait_until_stopped(self.mc)
            logger.info("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")

            await asyncio.to_thread(self.mc.send_angles, ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            await wait_until_stopped(self.mc)
            logger.info("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")
            
            await asyncio.to_thread(sync_flush_camera_buffer, self.cap, 15)
            await asyncio.sleep(0.5)

            # 5. Place 목표 좌표 계산 (Vision-Guided)
            print(f"\n🚀 Place 작업 시작: Vision-Guided 목표 좌표 계산 시작")
            place_calc_success, final_place_coords = await place_coords_calculator(self.cap)
            
            if not place_calc_success:
                logger.error("❌ Place 목표 좌표 계산 실패. Place 동작 중단.")
                # 미션 실패로 처리하지 않고, 현재 위치에서 그리퍼를 여는 등 안전 조치 필요
                await asyncio.to_thread(self.mc.send_angles, ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED) # 안전 포즈로 복귀
                await wait_until_stopped(self.mc)
                return 
            
            # 6. Place 동작 실행
            
            # Place 목표의 안전 포즈 (Z축 + 50)
            safe_place_coords = list(GLOBAL_TARGET_TMP_COORDS)
            
            # 안전 포즈로 이동
            logger.info(f"⬆️ Place 안전 포즈로 이동: X:{safe_place_coords[0]:.2f}, Y:{safe_place_coords[1]:.2f} (Z:{safe_place_coords[2]:.2f})")
            print(safe_place_coords)
            await asyncio.to_thread(self.mc.send_coords, safe_place_coords, MOVEMENT_SPEED - 30)
            await wait_until_stopped(self.mc)

            # Place 지점으로 하강
            logger.info(f"⬇️ Place 지점으로 하강: X:{final_place_coords[0]:.2f}, Y:{final_place_coords[1]:.2f} (Z:{final_place_coords[2]:.2f})")
            print(final_place_coords)
            await asyncio.to_thread(self.mc.send_coords, final_place_coords, MOVEMENT_SPEED - 30)
            await wait_until_stopped(self.mc)

            print("✊ 그리퍼 여는 중 (Place 동작)...")
            # 그리퍼 열기 (Place)
            await asyncio.to_thread(self.mc.set_gripper_value, GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            await asyncio.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ Place 완료 (그리퍼 열림).")

            # Place 완료 후 안전 포즈로 복귀
            await asyncio.to_thread(self.mc.send_coords, safe_place_coords, MOVEMENT_SPEED)
            await wait_until_stopped(self.mc)
            
        elif command == "Ready":
            logger.info("-> MyCobot: Ready 상태 수신, 대기 중...")
            
        else:
            logger.warning(f"-> MyCobot: 알 수 없는 명령: {command}")

async def arm_subscriber():
    """ OPC UA 클라이언트를 실행하고 구독을 설정하는 메인 함수 """
    global mc, cap, ai_model

    # 📌 1. AI 모델 로드
    ai_model = load_model(MODEL_WEIGHTS_PATH, NUM_CLASSES)
    
    # 📌 2. MyCobot 연결 초기화
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.set_color(0, 0, 255) 
        logger.info(f"MyCobot320이 {PORT}에 {BAUD} 속도로 성공적으로 연결되었습니다.")
        
        # 그리퍼 초기화 로직
        mc.set_gripper_mode(0)
        mc.init_electric_gripper()
        await asyncio.sleep(2)
        mc.set_electric_gripper(0)
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED, 1) # GRIPPER_OPEN_VALUE (85)로 열림
        await asyncio.sleep(2)
        logger.info(f"-> MyCobot320: 전기 그리퍼 초기화 완료 ({GRIPPER_OPEN_VALUE} 위치로 이동).")
        
    except Exception as e:
        logger.error(f"MyCobot320 연결 또는 초기화 실패: {e}")
        mc = None

    # 📌 3. 카메라 연결 초기화
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error(f"카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        cap = None

    if mc is None or cap is None:
        logger.error("MyCobot 또는 카메라 연결 문제로 Vision-Pick 미션을 수행할 수 없습니다.")
        if mc is not None:
            mc.close()
        return # 연결 실패 시 종료

    # 📌 4. OPC UA 연결 및 구독 설정
    logger.info(f"OPC UA 수신 서버에 연결 시도: {OPCUA_READ_URL}")

    try:
        async with AsyncuaClient(OPCUA_READ_URL) as client:
            logger.info("OPC UA 수신 서버에 성공적으로 연결되었습니다.")

            handler = SubHandler(mc, cap, ai_model)
            # 구독 간격 100ms
            sub = await client.create_subscription(100, handler)

            cmd_node = client.get_node(READ_METHOD_NODE_ID) 

            await sub.subscribe_data_change(cmd_node)
            logger.info(f"노드 '{READ_METHOD_NODE_ID}' 구독 시작. 데이터 수신 대기 중...")
            while True:
                await asyncio.sleep(1) # 클라이언트 유지
    
    except Exception as e:
        logger.error(f"OPC UA 연결 또는 구독 중 오류 발생: {e}")
    finally:
        # 📌 5. 자원 해제
        if mc is not None:
            mc.set_color(0, 0, 0)
            mc.close()
            logger.info("MyCobot 정리 완료.")
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("카메라 정리 완료.")
        logger.info("OPC UA 클라이언트 종료.")


if __name__ == "__main__":
    try:
        # 비동기 메인 루프 실행
        asyncio.run(arm_subscriber())
    except KeyboardInterrupt:
        logger.info("사용자 중단 (Ctrl+C). 프로그램 종료.")
    except Exception as e:
        logger.critical(f"프로그램 최종 오류: {e}")