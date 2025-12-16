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
SEQUENTIAL_MOVE_DELAY = 1.5
GRIPPER_ACTION_DELAY = 1

CAMERA_INDEX = 0
roi_start = (80, 30)
roi_end = (340, 400)
TARGET_CENTER_U = 210
TARGET_CENTER_V = 215

PIXEL_TO_MM_X = 0.526
PIXEL_TO_MM_Y = -0.698

MAX_PIXEL_ERROR = 5
PICK_Z_HEIGHT = 250

GRIPPER_OPEN_VALUE = 85
GRIPPER_CLOSED_VALUE = 25

LOWER_HSV = np.array([0, 0, 0])
UPPER_HSV = np.array([179, 255, 190])

CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90]

INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

TEST_PICK_POSE_WIDTH = [-237.90, 20, 183.6, -174.98, 0, 0]
TEST_PICK_POSE_HEIGHT = [-237.90, 20, 183.6, -174.98, 0, 90]

# --- 🎯 OPC UA 수신/송신 설정 (수정된 부분) ---
OPCUA_READ_URL = "opc.tcp://172.30.1.61:0630/freeopcua/server/"
OPCUA_WRITE_URL = "opc.tcp://172.30.1.61:0630/freeopcua/server/"

# 📌 읽기(구독) 노드 ID
READ_OBJECT_NODE_ID = "ns=2;i=3" # 구독 시 사용하지 않음 (CMD_NODE_PATH로 대체됨)
READ_METHOD_NODE_ID = "ns=2;s=read_arm_go_move"

# 📌 쓰기(Method Call) 노드 ID
WRITE_OBJECT_NODE_ID = "ns=2;i=3" # 미션 상태/비전 결과 전송 시 사용
WRITE_METHOD_NODE_ID = "ns=2;s=write_send_arm_json" # 비전 결과 전송 Method ID

# 📌 미션 상태 응답 Method ID (임시: 사용자 코드에는 없지만, 응답을 위해 13번을 가정하거나, 24번 재사용)
# send_mission_state가 write_vision_result와 동일한 노드를 사용하는 것으로 가정하고,
# 노드 ID를 명확히 분리하여 사용합니다.

# CMD_NODE_PATH = [
#     "0:Objects",
#     "2:ARM",
#     "2:read_arm_go_move"
# ]

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
# 🛠️ 로봇 및 비전 제어 함수 (기존과 동일)
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
    """
    물체 중심 좌표를 찾고, 물체 영역을 크롭한 이미지를 함께 반환합니다.
    """
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


async def pick_and_place_vision_guided(mc, cap, ai_model):
    """
    Vision-Guided 픽업 로직에 AI 객체 분류 기능을 수행하고 결과를 OPC UA로 전송합니다.
    (비동기 환경에 맞춰 time.sleep -> asyncio.sleep으로 수정 필요)
    """
    global SEQUENTIAL_MOVE_DELAY, MOVEMENT_SPEED, GRIPPER_OPEN_VALUE, GRIPPER_CLOSED_VALUE, GRIPPER_SPEED, GRIPPER_ACTION_DELAY, TEST_PICK_POSE_WIDTH, TEST_PICK_POSE_HEIGHT, transform

    # 현재 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        logger.error("❌ 카메라 프레임 읽기 실패. 픽업 중단.")
        # 실패 보고 및 초기 좌표 전송
        await send_mission_state("arm_mission_failure")
        await send_vision_result(module_type="Unknown", confidence=0.0, pick_coord=[0.0]*6)
        return False, "Unknown", 0.0, [0.0]*6
    
    center_x, center_y, largest_contour, rect, cropped_img = find_object_center(frame)

    if rect is None:
        logger.error("❌ 물체를 찾을 수 없습니다. 픽업 중단.")
        # 실패 보고 및 초기 좌표 전송 (실패 시)
        await send_mission_state("arm_mission_failure")
        await send_vision_result(module_type="Unknown", confidence=0.0, pick_coord=[0.0]*6)
        return False, "Unknown", 0.0, [0.0]*6
        
    (center_u, center_v), (w, h), angle = rect
    
    # 📌 AI 객체 분류 수행
    predicted_class, confidence = classify_object(ai_model, transform, cropped_img)
    
    print(f"\n🧠 AI 분류 결과: **{predicted_class}** (신뢰도: {confidence*100:.2f}%)")
    
    # 픽업 자세 결정 (가로/세로)
    if w > h:
        target_pose = list(TEST_PICK_POSE_WIDTH)
        logger.info(f"📐 물체 장축: 가로. 최종 Pose: TEST_PICK_POSE_WIDTH 선택.")
    else: 
        target_pose = list(TEST_PICK_POSE_HEIGHT)
        logger.info(f"📐 물체 장축: 세로. 최종 Pose: TEST_PICK_POSE_HEIGHT 선택.")
        
    # ----------------------------------------------------
    # 로봇 이동 시작 (동작 시작 전에 분류 결과 전송하는 것이 일반적)
    # ----------------------------------------------------
    
    # 사용자 지정 crop 후 server로 송신
    send_img = frame[30:400, 30:340]
    
    # OPC UA 결과 전송 (로봇 동작 직전에 전송)
    await send_vision_result(
        module_type=predicted_class, 
        confidence=confidence, 
        pick_coord=target_pose,
        image_to_send=send_img
    )

    # 로봇 이동 (time.sleep을 asyncio.sleep으로 대체)
    safe_pose = list(target_pose)
    safe_pose[2] += 50 
    
    mc.send_coords(safe_pose, MOVEMENT_SPEED)
    await asyncio.sleep(SEQUENTIAL_MOVE_DELAY)

    logger.info(f"\n⬇️ 픽업 시작: X:{target_pose[0]:.2f}, Y:{target_pose[1]:.2f} (Z:{target_pose[2]:.2f}) 하강.")
    mc.send_coords(target_pose, MOVEMENT_SPEED - 30)
    await asyncio.sleep(SEQUENTIAL_MOVE_DELAY)
    
    mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
    await asyncio.sleep(GRIPPER_ACTION_DELAY)
    
    target_pose[2] += 100
    mc.send_coords(target_pose, MOVEMENT_SPEED)
    await asyncio.sleep(SEQUENTIAL_MOVE_DELAY)
    
    logger.info("✅ 픽업 및 안전 높이 복귀 완료.")
    
    return True, predicted_class, confidence, target_pose

# ===============================================
# 🌐 OPC UA 통신 함수 (READ/WRITE NODE ID 분리하여 수정)
# ===============================================

async def send_mission_state(status: str):
    """미션 상태(arm_mission_success/failure)를 서버에 송신합니다. (WRITE_OBJECT_NODE_ID, WRITE_METHOD_NODE_ID 사용)"""
    # ⚠️ 미션 상태 전송을 위한 별도의 Method ID가 없으므로
    # READ_METHOD_NODE_ID("ns=2;i=13")를 사용하여 Call하는 것으로 가정합니다.
    
    global OPCUA_WRITE_URL, WRITE_OBJECT_NODE_ID, WRITE_METHOD_NODE_ID 

    # mission_state = { "status": status }
    mission_state = { 
        "module_type": "Mission_State",
        "classification_confidence": 0.0,
        "pick_coord": ["0.00", "0.00", "0.00", "0.00", "0.00", "0.00"],
        "pick_coord_confidence": 0.0,
        "img": "",
        "status": status # 미션 상태는 status 필드에 담아 전송
    }

    json_str = json.dumps(mission_state)
    
    logger.info(f"OPC UA 미션 상태 송신 서버에 연결 시도: {OPCUA_WRITE_URL}")
    try:
        async with AsyncuaClient(OPCUA_WRITE_URL) as client:
            obj = client.get_node(WRITE_OBJECT_NODE_ID)
            # ⚠️ 이 부분에서 READ_METHOD_NODE_ID를 사용하도록 수정했습니다.
            method_node = client.get_node(WRITE_METHOD_NODE_ID)
            
            print(f"\n[OPC UA WRITE - MISSION_STATE] call_method(status='{status}') (Method: {WRITE_METHOD_NODE_ID})")
            json_variant = ua.Variant(json_str, ua.VariantType.String)

            result_code, result_message = await obj.call_method(
                method_node.nodeid,
                json_variant
            )
            logger.info(f"OPC UA 미션 상태 송신 완료. ResultCode: {result_code}")
            return result_code, result_message

    except Exception as e:
        logger.error(f"OPC UA 미션 상태 송신 중 오류 발생: {e}")
        return -1, str(e)


async def send_vision_result(module_type: str, confidence: float, pick_coord: list, image_to_send: np.ndarray = None):
    """
    분류 및 픽업 결과를 JSON 형태로 묶어 OPC UA 서버에 송신합니다. (WRITE_OBJECT_NODE_ID, WRITE_METHOD_NODE_ID 사용)
    """
    global OPCUA_WRITE_URL, WRITE_OBJECT_NODE_ID, WRITE_METHOD_NODE_ID
    
    # --- 📌 이미지 처리 및 인코딩 로직 (기존과 동일) ---
    base64_img_str = ""
    if image_to_send is not None:
        try:
            # 1. 해상도 축소 (예: 224x224로 리사이즈)
            resized_img = cv2.resize(image_to_send, (224, 224), interpolation=cv2.INTER_AREA)

            # 2. JPEG 압축 인코딩 (압축 품질 80 설정)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80] 
            _, buffer = cv2.imencode('.jpg', resized_img, encode_param)
            
            # 3. Base64 인코딩 (바이너리 데이터를 ASCII 문자열로 변환)
            base64_img_bytes = base64.b64encode(buffer)
            base64_img_str = base64_img_bytes.decode('utf-8')
            logger.info(f"🖼️ 이미지 인코딩 완료. Base64 문자열 길이: {len(base64_img_str)}")
            
        except Exception as e:
            logger.error(f"이미지 인코딩 중 오류 발생: {e}")
            base64_img_str = "" # 오류 발생 시 이미지 필드를 빈 문자열로 둠
    
    # 비전 결과 JSON 데이터 구성
    vision_result = {
        "module_type": module_type,
        "classification_confidence": confidence,
        "pick_coord": [f"{c:.2f}" for c in pick_coord], # 로봇 좌표를 문자열 리스트로 변환하여 전송
        "pick_coord_confidence": 0.9984073221683503,
        "img": base64_img_str
    }
    json_str = json.dumps(vision_result)

    # 📌 추가된 부분: 클라이언트가 실제로 전송하는 JSON 문자열 출력
    print("\n========================================================")
    print(f"🚀 [VISION RESULT] 클라이언트가 서버로 전송하는 최종 JSON:")
    print(json_str)
    print("========================================================\n")
    
    try:
        async with AsyncuaClient(OPCUA_WRITE_URL) as client:
            # ⚠️ 이 부분에서 WRITE_OBJECT_NODE_ID 사용
            obj = client.get_node(WRITE_OBJECT_NODE_ID)
            # ⚠️ 이 부분에서 WRITE_METHOD_NODE_ID 사용
            method_node = client.get_node(WRITE_METHOD_NODE_ID)

            # 📌 개선: 문자열을 ua.Variant(ua.String)으로 명시적 변환하여 전송
            json_variant = ua.Variant(json_str, ua.VariantType.String)

            print(f"\n[OPC UA WRITE - VISION_RESULT] call_method(Module: {module_type}, Conf: {confidence*100:.2f}%) (Method: {WRITE_METHOD_NODE_ID})")
            result_code, result_message = await obj.call_method(
                method_node.nodeid,
                json_variant # ua.Variant 객체 전송
            )
            logger.info(f"OPC UA 비전 결과 송신 완료. ResultCode: {result_code}")

    except Exception as e:
        logger.error(f"OPC UA 비전 결과 송신 중 오류 발생: {e}")


# ----------------------
# OPC UA DataChange 구독 핸들러 클래스 (기존과 동일)
# ----------------------
class SubHandler:
    
    def __init__(self, mycobot_instance, camera_instance, ai_model_instance):
        self.mc = mycobot_instance
        self.cap = camera_instance
        self.ai_model = ai_model_instance
        logger.info("SubHandler 초기화 완료.")
        

    def datachange_notification(self, node, val, data):
        """데이터 변경 알림 시 호출되는 비동기적 콜백 함수"""
        # 비동기 함수인 execute_command_and_respond를 별도의 태스크로 실행
        # 로봇/비전 작업은 시간이 오래 걸리므로 콜백이 빨리 끝나도록 태스크로 만듭니다.
        asyncio.create_task(self.execute_command_and_respond(val))

    async def execute_command_and_respond(self, val):
        """명령을 파싱하고 MyCobot 동작을 수행한 후 응답합니다."""
        
        print(f"\n[OPC UA READ] 수신 값: {val}")

        command = None
        if isinstance(val, str):
            try:
                json_data = json.loads(val)
                if "move_command" in json_data:
                    command = json_data["move_command"]
            except json.JSONDecodeError:
                command = val # Ready 같은 일반 문자열도 command로 간주

        
        # 3. MyCobot 동작 수행 및 응답
        if command and self.mc is not None and self.cap is not None:
            
            if command == "go_home":
                # 1번 키와 같은 동작: CONVEYOR_CAPTURE_POSE로 이동
                logger.info("-> MyCobot: go_home 명령 수행 (CONVEYOR_CAPTURE_POSE로 이동)")
                self.mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                await asyncio.sleep(SEQUENTIAL_MOVE_DELAY)
                self.mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
                await asyncio.sleep(SEQUENTIAL_MOVE_DELAY)
                
                # 동작 완료 보고
                await send_mission_state("arm_mission_success")
            
            elif command == "mission_start":
                # 4번 키와 같은 동작: Vision-Guided Pick 수행
                logger.info("-> MyCobot: mission_start 명령 수행 (Vision-Guided Pick 시작)")
                
                # --- 미션 시작 동작 ---
                success, module_type, confidence, pick_coord = await pick_and_place_vision_guided(self.mc, self.cap, self.ai_model)
                
                # --- 미션 종료 ---
                if success:
                    logger.info("-> MyCobot: Vision-Guided Pick 완료. OPC UA 응답 송신 시작.")
                    await send_mission_state("arm_mission_success")
                else:
                    logger.error("-> MyCobot: Vision-Guided Pick 실패. OPC UA 실패 보고 송신.")
                    await send_mission_state("arm_mission_failure")
                    
            elif command == "Ready":
                logger.info("-> MyCobot: Ready 상태 수신, 대기 중...")
                
            else:
                logger.warning(f"-> MyCobot: 알 수 없는 명령: {command}")
        elif command and (self.mc is None or self.cap is None):
            logger.warning(f"-> MyCobot 또는 카메라 연결 문제로 '{command}' 명령을 수행할 수 없습니다.")


async def arm_subscriber():
    """
    OPC UA 클라이언트를 실행하고 구독을 설정하는 메인 함수
    """
    global mc, cap, ai_model

    # 📌 1. AI 모델 로드 (MyCobot/Camera 전에 수행)
    ai_model = load_model(MODEL_WEIGHTS_PATH, NUM_CLASSES)
    
    # 📌 2. MyCobot 연결 초기화
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.set_color(0, 0, 255) 
        logger.info(f"MyCobot320이 {PORT}에 {BAUD} 속도로 성공적으로 연결되었습니다.")
        
        # 그리퍼 초기화 로직
        mc.set_gripper_mode(0)
        mc.init_electric_gripper()
        time.sleep(2)
        mc.set_electric_gripper(0)
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED, 1) # GRIPPER_OPEN_VALUE (85)로 열림
        time.sleep(2)
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