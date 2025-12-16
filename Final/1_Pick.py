import torch
import cv2
import time
import os
import sys
import csv
import numpy as np
from pymycobot import MyCobot320
from torchvision import transforms
from PIL import Image

# ===============================================
# 📌 AI 모델 설정 (추가된 부분)
# ===============================================
CLASS_NAMES = ["ESP32", "L298N(Motor)", "MB102(Power)"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOBILENET_MEAN = [0.485, 0.456, 0.406]
MOBILENET_STD = [0.229, 0.224, 0.225]
# ⚠️ 모델 파일 경로를 실제 파일 이름으로 변경해주세요.
MODEL_WEIGHTS_PATH = "checkpoint_mobilenetv3_classifier_e5_acc1.0000.pth"

# ===============================================
# ⚙️ MyCobot 및 비전 시스템 설정 (기존 설정)
# ===============================================
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
ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90]

INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

TEST_PICK_POSE_WIDTH = [-229.30, 20, 183.6, -174.98, 0, 0]
TEST_PICK_POSE_HEIGHT = [-229.30, 7.80, 183.6, -174.98, 0, 90]

DATA_DIR = "capture"
CSV_FILE = os.path.join(DATA_DIR, "pixel_to_mm_data.csv")
COORDINATE_FILE = "pick_coordinate.txt"

# ===============================================
# 🧠 AI 모델 함수 (추가된 부분)
# ===============================================

# 이미지 전처리 (학습 시 사용한 정규화/리사이즈와 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MOBILENET_MEAN, std=MOBILENET_STD)
])

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
        print(f"\n❌ 오류: 모델 가중치 파일({model_path})을 찾을 수 없습니다. 경로를 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 모델 로드 중 예기치 않은 오류 발생: {e}")
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
# 🛠️ 로봇 및 비전 제어 함수 (수정된 부분 포함)
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
    물체 중심 좌표를 찾고, 물체 영역을 크롭한 이미지를 함께 반환합니다. (수정됨)
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
    
    # 디버깅용 마스크 화면 표시
    cv2.imshow('Masked (Final Target)', final_mask)
    
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

def pick_and_place_vision_guided(mc, cap, frame, ai_model):
    """
    Vision-Guided 픽업 로직에 AI 객체 분류 기능을 추가합니다. (수정됨)
    """
    global SEQUENTIAL_MOVE_DELAY, MOVEMENT_SPEED, GRIPPER_OPEN_VALUE, GRIPPER_CLOSED_VALUE, GRIPPER_SPEED, GRIPPER_ACTION_DELAY, TEST_PICK_POSE_WIDTH, TEST_PICK_POSE_HEIGHT

    center_x, center_y, largest_contour, rect, cropped_img = find_object_center(frame) # 크롭된 이미지 수신

    if rect is None:
        print("❌ 물체를 찾을 수 없습니다. 픽업 중단.")
        return False
        
    (center_u, center_v), (w, h), angle = rect
    
    # 📌 AI 객체 분류 수행 (추가된 부분)
    predicted_class, confidence = classify_object(ai_model, transform, cropped_img)
    
    print(f"\n🧠 AI 분류 결과: **{predicted_class}** (신뢰도: {confidence*100:.2f}%)")
    
    # 픽업 자세 결정 (기존 로직 유지)
    if w > h:
        target_pose = list(TEST_PICK_POSE_WIDTH)
        print(f"📐 물체 장축: 가로 (w={w:.2f} > h={h:.2f}). 최종 Pose: TEST_PICK_POSE_WIDTH 선택.")
    else: 
        target_pose = list(TEST_PICK_POSE_HEIGHT)
        print(f"📐 물체 장축: 세로 (h={h:.2f} >= w={w:.2f}). 최종 Pose: TEST_PICK_POSE_HEIGHT 선택.")
        
    # 픽셀-로봇 좌표 변환 및 오차 계산 (기존 로직 유지)
    delta_X, delta_Y, delta_u_pixel, delta_v_pixel = convert_pixel_to_robot_move(center_x, center_y)
    error = np.sqrt(delta_u_pixel**2 + delta_v_pixel**2)
    
    print(f"🔍 픽셀 오차: {error:.2f} 픽셀. 로봇 보정 이동량: (X: {delta_X:.2f}mm, Y: {delta_Y:.2f}mm)")
    
    # 픽업 좌표 보정
    target_pose[0] += delta_X
    target_pose[1] += delta_Y
    
    # 로봇 이동 (기존 로직 유지)
    safe_pose = list(target_pose)
    safe_pose[2] += 50 
    
    mc.send_coords(safe_pose, MOVEMENT_SPEED)
    time.sleep(SEQUENTIAL_MOVE_DELAY)

    print(f"\n⬇️ 픽업 시작: X:{target_pose[0]:.2f}, Y:{target_pose[1]:.2f} (Z:{target_pose[2]:.2f}) 하강.")
    mc.send_coords(target_pose, MOVEMENT_SPEED - 30)
    time.sleep(SEQUENTIAL_MOVE_DELAY)
    
    mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
    time.sleep(GRIPPER_ACTION_DELAY)
    
    target_pose[2] += 100
    mc.send_coords(target_pose, MOVEMENT_SPEED)
    time.sleep(SEQUENTIAL_MOVE_DELAY)
    
    print("✅ 픽업 및 안전 높이 복귀 완료.")
    
    # 📌 분류 결과에 따라 다음 동작을 수행하는 로직을 추가할 수 있습니다.
    # 예: if predicted_class == "ESP32": place_at_A() 
    #     elif predicted_class == "L298N(Motor)": place_at_B()
    
    return True

def load_and_move_coords(mc, file_path):
    global MOVEMENT_SPEED, SEQUENTIAL_MOVE_DELAY
    
    print(f"\n📁 {file_path} 파일에서 좌표 로딩 시작...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            coords_str = content.strip('[]').split(', ')
            
            target_coords = [float(x) for x in coords_str if x]
            
            if len(target_coords) == 6:
                print(f"✅ 좌표 로딩 성공: {target_coords}")
                
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                
                mc.send_coords(target_coords, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                
                print("🚀 파일에서 로딩된 좌표로 이동 완료.")
            else:
                print(f"❌ 오류: 파일 내용이 6개의 좌표가 아닙니다. 내용: {content}")
                
    except FileNotFoundError:
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다. 파일을 생성해주세요.")
    except ValueError as e:
        print(f"❌ 오류: 파일 내용 변환 중 문제 발생 (숫자 형식 확인 필요). 오류: {e}")
    except Exception as e:
        print(f"❌ 로봇 이동 중 통신 오류 발생: {e}")

# ===============================================
# 🚀 메인 실행 함수 (AI 모델 로드 추가)
# ===============================================

def main():
    
    # 📌 AI 모델 로드 (추가된 부분)
    ai_model = load_model(MODEL_WEIGHTS_PATH, NUM_CLASSES)
    
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.power_on()
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 초기 상태: 파워 ON (서보 잠금)")

        mc.set_gripper_mode(0)
        mc.init_electric_gripper()
        time.sleep(2)
        mc.set_electric_gripper(0)
        
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
        time.sleep(GRIPPER_ACTION_DELAY)
        print(f"✅ 그리퍼 초기화 완료. 위치: **{GRIPPER_OPEN_VALUE} (열림)**.")
        
    except Exception as e:
        print(f"\n❌ MyCobot 연결 실패 ({PORT}): {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"\n❌ 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        mc.close()
        sys.exit(1)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Target_Center_U', 'Target_Center_V', 'Robot_Coord_X', 'Robot_Coord_Y'])
        print(f"✅ 데이터 기록 파일 생성 완료: {CSV_FILE}")

    last_center_u = None
    last_center_v = None

    print(f"✅ 현재 카메라 창 크기: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} 픽셀")
    print("\n--- 🔑 로봇 제어 가이드 ---")
    print(" [q]: 종료 | [s]: 티칭 시작(서보 해제) | [e]: 티칭 종료(서보 잠금)")
    print(" [0]: 0도 자세 | [1]: 컨베이어 캡처 자세 | [2]: 픽업 자세 (테스트)")
    print(" [3]: 로봇팔 위 캡처 자세 | [4]: Vision-Guided 픽업 | [5]: 기준 좌표 이동")
    print(" [j]: Joint 값 읽기 | [k]: Coordinates 읽기 | [g/h]: 그리퍼 닫기/열기")
    print(" [c]: 현재 화면 캡처 및 좌표 기록")
    print(f" [r]: {COORDINATE_FILE} 파일의 좌표 로드 및 이동")
    print(" [w/x]: X+1mm / X-1mm 이동 | [d/a]: Y+1mm / Y-1mm 이동")
    print("----------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # 📌 find_object_center 함수에서 크롭된 이미지를 함께 수신 (수정됨)
        center_x, center_y, largest_contour, rect, cropped_img = find_object_center(frame.copy())
        
        # ... (나머지 시각화 로직은 유지) ...

        roi_center_x, roi_center_y = (roi_start[0] + roi_end[0]) // 2, (roi_start[1] + roi_end[1]) // 2
        cv2.rectangle(frame, roi_start, roi_end, (255, 255, 255), 2)
        cv2.circle(frame, (roi_center_x, roi_center_y), 5, (0, 0, 0), -1) 
        cv2.putText(frame, "ROI / Target", (roi_center_x + 10, roi_center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if center_x is not None:
            last_center_u, last_center_v = center_x, center_y
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1) 
            
            # 📌 시각화에 AI 분류 결과 추가
            predicted_class, confidence = classify_object(ai_model, transform, cropped_img)
            
            cv2.putText(frame, f"Class: {predicted_class}", 
                        (roi_center_x - 200, roi_center_y + 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Conf: {confidence*100:.2f}%", 
                        (roi_center_x - 200, roi_center_y + 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Detected U(X): {center_x}, Detected V(Y): {center_y}", 
                        (roi_center_x - 200, roi_center_y + 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Diff. U(X): {(roi_center_x-center_x)}, Diff. V(Y): {(roi_center_y-center_y)}", 
                        (roi_center_x - 200, roi_center_y + 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "Target Not Found", (roi_center_x - 310, roi_center_y + 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('MyCobot Pick Task', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n👋 프로그램 종료 명령 수신. 자원을 해제합니다...")
            break
            
        elif key == ord('r'):
            load_and_move_coords(mc, COORDINATE_FILE)
            
        elif key == ord('4'):
            print("\n✨ **Vision-Guided Pick Task 시작**")
            ret, current_frame = cap.read()
            if ret:
                # 📌 AI 모델을 인수로 전달 (수정됨)
                success = pick_and_place_vision_guided(mc, cap, current_frame, ai_model) 
                if success:
                    print("👍 픽업 태스크 성공적으로 완료.")
                else:
                    print("😭 픽업 태스크 실패.")
            else:
                print("❌ 카메라 프레임 읽기 실패.")
                
        # (나머지 로직... 키 입력, 로봇 제어, 좌표 기록 등)

        elif key == ord('s'):
            print("\n▶️ **티칭 모드 시작** (모든 서보 잠금 해제, 수동 제어 가능)")
            mc.release_all_servos()
            
        elif key == ord('e'):
            print("\n⏸️ **티칭 모드 종료** (모든 서보 잠금, 로봇 움직임 고정)")
            mc.power_on()

        elif key in [ord('w'), ord('x'), ord('a'), ord('d')]:
            current_coords = mc.get_coords()
            
            if not isinstance(current_coords, list) or all(c == -1 for c in current_coords):
                current_coords = list(TEST_PICK_POSE_WIDTH)
                print("⚠️ 로봇 좌표를 읽을 수 없어 기준 좌표를 사용합니다.")
            else:
                current_coords = list(current_coords) 
            
            move_x, move_y = 0.0, 0.0
            axis_name = ""
            
            if key == ord('w'):
                move_x = 5
                axis_name = "X + 5mm"
            elif key == ord('x'):
                move_x = -5
                axis_name = "X - 5mm"
            elif key == ord('d'): 
                move_y = 5
                axis_name = "Y + 5mm"
            elif key == ord('a'): 
                move_y = -5
                axis_name = "Y - 5mm"
            
            if axis_name:
                current_coords[0] += move_x
                current_coords[1] += move_y
                
                mc.send_coords(current_coords, MOVEMENT_SPEED - 30)
                time.sleep(0.1)
                
                print(f"en➡️ 증분 이동 ({axis_name}): 새로운 좌표 (X:{current_coords[0]:.2f}, Y:{current_coords[1]:.2f})")

        elif key == ord('0'):
            print(f"\n🔄 로봇을 0도 자세 이동 시작...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED) 
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            print("✅ 0도 자세 이동 완료.")
        
        elif key == ord('1'):
            print(f"\n🚀 컨베이어 캡처 자세 ({CONVEYOR_CAPTURE_POSE})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")
            
        elif key == ord('2'):
            print(f"\n⬇️ 테스트 픽업 가로 자세 ({TEST_PICK_POSE_WIDTH})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_coords(TEST_PICK_POSE_WIDTH, MOVEMENT_SPEED) 
            print("✅ TEST_PICK_POSE_WIDTH 이동 완료.")
        
        elif key == ord('3'):
            print(f"\n⬇️ 테스트 픽업 세로 자세 ({TEST_PICK_POSE_HEIGHT})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_coords(TEST_PICK_POSE_HEIGHT, MOVEMENT_SPEED) 
            print("✅ TEST_PICK_POSE_HEIGHT 세로 이동 완료.")

        elif key == ord('c'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"
            save_path = os.path.join(DATA_DIR, filename)
            
            cv2.imwrite(save_path, frame)
            
            if last_center_u is not None:
                try:
                    current_coords = mc.get_coords()
                    if isinstance(current_coords, list) and not all(c == -1 for c in current_coords):
                        with open(CSV_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, last_center_u, last_center_v, current_coords[0], current_coords[1]])
                        print(f"\n📸 데이터 캡처 완료: {save_path}. 픽셀: ({last_center_u}, {last_center_v}), 로봇 X/Y: ({current_coords[0]:.2f}, {current_coords[1]:.2f})")
                    else:
                        print(f"\n❌ 로봇 좌표를 읽을 수 없어 픽셀 데이터만 캡처됨: {save_path}")
                        with open(CSV_FILE, 'a', newline='') as f:
                            csv.writer(f).writerow([timestamp, last_center_u, last_center_v, 'N/A', 'N/A'])
                except Exception as e:
                    print(f"\n❌ 로봇 통신 오류로 좌표 기록 실패: {e}")
            else:
                print(f"\n🔴 물체가 검출되지 않아 캡처만 저장됨: {save_path}")

        elif key == ord('j'):
            current_angles = mc.get_angles()
            if isinstance(current_angles, list) and not all(c == -1 for c in current_angles): 
                print(f"\n📐 현재 Joint 값 (J1~J6): **{current_angles}**")
            else:
                print("\n❌ Joint 값을 읽을 수 없습니다. 로봇 연결 상태를 확인하세요.")

        elif key == ord('k'):
            current_coords = mc.get_coords()
            if isinstance(current_coords, list) and not all(c == -1 for c in current_coords): 
                print(f"\n🗺️ 현재 Coordinates (X, Y, Z, R, P, Y): **{current_coords}**") 
            else:
                print("\n❌ Coordinates 값을 읽을 수 없습니다. 로봇 연결 상태를 확인하세요.")
        
        elif key == ord('g'):
            print("\n✊ 그리퍼 닫는 중...")
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED) 
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 닫힘 완료 (위치: **{GRIPPER_CLOSED_VALUE}**).")
            
        elif key == ord('h'):
            print("\n👐 그리퍼 여는 중...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 열림 완료 (위치: **{GRIPPER_OPEN_VALUE}**).")
        
    print("🧹 자원 해제 중: 카메라 및 로봇 연결 종료...")
    cap.release()
    cv2.destroyAllWindows()
    try:
        mc.close()
    except Exception:
        pass
    print("👍 프로그램 종료 완료.")

if __name__ == "__main__":
    # ⚠️ PyTorch 로드 시 GPU 문제나 환경 문제가 발생할 경우, 
    # torch.hub.load() 부분을 pip install torchvision을 통해 import한 
    # models.mobilenet_v3_small 로 대체하는 것을 고려해 보세요.
    main()