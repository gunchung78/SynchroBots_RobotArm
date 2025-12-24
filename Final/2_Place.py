import cv2
import time
import os
import sys
import csv
import numpy as np
from pymycobot import MyCobot320 # MyCobot 320 로봇 팔 제어 라이브러리

# ====================================================
# 1. 로봇 및 비전 설정 변수
# ====================================================

# --- 로봇 통신 및 속도 설정 ---
PORT = "COM3"             # 🖧 로봇 팔 통신 포트 (Windows 환경)
BAUD = 115200             # ⚡ 로봇 팔 통신 속도

MOVEMENT_SPEED = 70       # ⚙️ 관절/좌표 이동 속도 (퍼센트 단위, 1-100)
GRIPPER_SPEED = 50        # ⚙️ 그리퍼 작동 속도 (20 -> 50으로 상향 조정)
SEQUENTIAL_MOVE_DELAY = 1.5 # ⏱️ 자세 이동 명령 간 대기 시간 (안정성 확보를 위해 1.5초로 조정)
GRIPPER_ACTION_DELAY = 1  # ⏱️ 그리퍼 작동 후 대기 시간

# --- 카메라 및 ROI 설정 ---
CAMERA_INDEX = 0          # 📷 OpenCV 카메라 인덱스 (기본 카메라)
roi_start = (100, 50)        # 🔍 관심 영역(ROI) 시작점 (좌상단 픽셀 좌표)
roi_end = (340, 260)      # 🔍 관심 영역(ROI) 끝점 (우하단 픽셀 좌표)
TARGET_CENTER_U = 320     # 🎯 픽셀 추적 목표 U (X) 좌표 (ROI 중심)
TARGET_CENTER_V = 180     # 🎯 픽셀 추적 목표 V (Y) 좌표 (ROI 중심)

# --- 픽셀-MM 변환 비율 (Vision-Guided Control 핵심 파라미터) ---
# 로봇 1mm 움직일 때 픽셀 변화량. 측정 후 정확한 값으로 업데이트 필요.
PIXEL_TO_MM_X = 0.526      # X축 (U) 픽셀당 로봇 MM 변환 비율 [mm/pixel]
PIXEL_TO_MM_Y = -0.698      # Y축 (V) 픽셀당 로봇 MM 변환 비율 [mm/pixel]

# --- Vision-Guided 제어 파라미터 ---
MAX_PIXEL_ERROR = 5       # 정렬 종료 임계값 (5 픽셀 이내)
PICK_Z_HEIGHT = 260       # 픽업 시 최종 Z 좌표 (로봇의 Z축 위치)

# --- 그리퍼 값 설정 ---
GRIPPER_OPEN_VALUE = 85   # 👐 그리퍼 완전 열림 위치 값 (max 100)
GRIPPER_CLOSED_VALUE = 25 # ✊ 그리퍼 완전 닫힘 위치 값 (min 0)

# --- 색상 검출 (빨간색 HSV 범위) ---
LOWER_RED_HSV1 = np.array([0, 100, 100])
UPPER_RED_HSV1 = np.array([15, 255, 255])
LOWER_RED_HSV2 = np.array([155, 100, 100])
UPPER_RED_HSV2 = np.array([179, 255, 255])

# --- 주요 로봇 좌표 (Coords: X, Y, Z, Rx, Ry, Rz) ---
GLOBAL_TARGET_COORDS = [-114, -195, 250, 177.71, 0.22, 0]

# --- 주요 로봇 자세 (Joint Angles [J1, J2, J3, J4, J5, J6]) ---
CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90] # 컨베이어벨트 캡처를 위한 시야 확보 자세
ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90] # 로봇 팔 위 물체 캡처 자세

INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86] # 충돌 방지 경유 자세
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                      # 모든 관절 0도 자세

# 픽업/플레이스 테스트용 관절 자세 (경로 테스트용)
TMP_PICK_POSE = [-90, 10, 90, -20, -90, -90] 
TEST_PICK_POSE = [-90, 30, 90, -30, -90, -90]

# --- 데이터 저장 설정 ---
DATA_DIR = "capture" # 데이터 저장 디렉토리
CSV_FILE = os.path.join(DATA_DIR, "pixel_to_mm_data.csv") # 픽셀-로봇 좌표 기록 파일
COORDINATE_FILE = "place_coordinate.txt" # ✅ 좌표 로딩 파일 이름 정의

# ====================================================
# 2. Vision-Guided 제어를 위한 핵심 함수
# ====================================================

def convert_pixel_to_robot_move(current_center_u, current_center_v):
    """
    현재 검출된 중심 픽셀과 목표 픽셀 중심의 차이를
    로봇이 움직여야 할 실제 mm 단위의 이동 거리(Delta X, Delta Y)로 변환합니다.
    """
    global TARGET_CENTER_U, TARGET_CENTER_V, PIXEL_TO_MM_X, PIXEL_TO_MM_Y
    
    # 1. 픽셀 오차 계산 (검출된 위치 - 목표 위치)
    delta_u_pixel = current_center_u - TARGET_CENTER_U # U(X)축 픽셀 오차
    delta_v_pixel = current_center_v - TARGET_CENTER_V # V(Y)축 픽셀 오차
    
    # 2. MM으로 변환
    delta_X_mm = delta_u_pixel * PIXEL_TO_MM_X
    delta_Y_mm = delta_v_pixel * PIXEL_TO_MM_Y
    
    # 3. 로봇 이동 명령 (부호 조정)
    # 목표: 픽셀 오차의 반대 방향으로 로봇을 이동시켜 물체를 중앙으로 수렴시킵니다.
    final_delta_X = -delta_X_mm # X축 이동 거리 (픽셀 오차의 반대 방향)
    final_delta_Y = -delta_Y_mm # Y축 이동 거리 (픽셀 오차의 반대 방향)
    
    return final_delta_X, final_delta_Y, delta_u_pixel, delta_v_pixel

def find_red_center(frame):
    """ ROI 내부에서만 가장 큰 빨간색 영역의 중심을 찾습니다. """
    global roi_start, roi_end
    
    # 1. ROI 영역만 남기고 나머지는 검은색으로 지우는 마스크 생성
    # (Computer Vision 전문가용 팁: Bitwise 연산을 활용한 전처리)
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # ROI 영역(y1:y2, x1:x2)만 255(흰색)로 채웁니다.
    roi_mask[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = 255
    
    # 2. 원본 이미지와 ROI 마스크를 합쳐서 관심 영역만 추출
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
    # 3. 추출된 영역(masked_frame)에서만 HSV 색상 검출 수행
    hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv_frame, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask2 = cv2.inRange(hsv_frame, LOWER_RED_HSV2, UPPER_RED_HSV2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 노이즈 제거 (Opening/Closing)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # 4. 윤곽선 찾기
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50: 
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                return (center_x, center_y, largest_contour)
                
    return (None, None, None)

def align_to_target(mc, cap):
    """ Vision-Guided Control (Single-Shot): 검출된 빨간색 구역으로 로봇 팔이 좌표 이동(send_coords) 시킵니다. """
    global GLOBAL_TARGET_COORDS, MOVEMENT_SPEED, PICK_Z_HEIGHT
    
    # 1. 이미지 캡처 및 중심 찾기
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 수신 실패. 정렬 중지.")
        return False

    center_u, center_v, _ = find_red_center(frame)
    
    if center_u is None:
        print(f"🔴 빨간색 물체 미검출. 정렬 중지.")
        return False

    # 2. 오차 계산 및 MM 변환
    delta_X_mm, delta_Y_mm, delta_u_pixel, delta_v_pixel = convert_pixel_to_robot_move(center_u, center_v)
    
    total_pixel_error = np.sqrt(delta_u_pixel**2 + delta_v_pixel**2)
    
    print(f"\n--- 🤖 Vision-Guided 정렬 (Single Shot) ---")
    print(f"  [Detect] 픽셀 오차: {total_pixel_error:.2f}px (U: {delta_u_pixel}, V: {delta_v_pixel})")
    print(f"  [Move] 이동 명령: X:{delta_X_mm:.2f}mm, Y:{delta_Y_mm:.2f}mm")

    # 3. 최종 목표 좌표 계산
    current_coords = list(GLOBAL_TARGET_COORDS) # 기준 좌표 복사
    
    # 픽셀 오차를 MM으로 변환한 만큼 로봇 좌표에 추가하여 '정렬된' 목표 좌표를 생성
    current_coords[0] += delta_X_mm # X축 이동 명령 적용
    current_coords[1] += delta_Y_mm # Y축 이동 명령 적용
    
    # Z축 높이는 미리 설정된 픽업 높이로 고정
    current_coords[2] = PICK_Z_HEIGHT 

    # 4. 로봇 이동 명령 실행
    mc.send_coords(current_coords, MOVEMENT_SPEED-50)
    time.sleep(SEQUENTIAL_MOVE_DELAY) 

    print(f"✅ 정렬 및 이동 완료. 최종 좌표: X:{current_coords[0]:.2f}, Y:{current_coords[1]:.2f}, Z:{PICK_Z_HEIGHT:.2f}")
    
    return True

# ====================================================
# 4. 좌표 파일 로딩 및 이동 함수 (R 키 기능)
# ====================================================

def load_and_move_coords(mc, file_path):
    """ place_coordinate.txt 파일에서 좌표를 읽어와 로봇 팔을 이동시킵니다. """
    global MOVEMENT_SPEED, SEQUENTIAL_MOVE_DELAY
    
    print(f"\n📁 {file_path} 파일에서 좌표 로딩 시작...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # 문자열에서 [ ]와 공백 제거 후 쉼표로 분리
            coords_str = content.strip('[]').split(', ')
            
            # 문자열 리스트를 float 리스트로 변환
            target_coords = [float(x) for x in coords_str if x]
            
            if len(target_coords) == 6:
                print(f"✅ 좌표 로딩 성공: {target_coords}")
                
                # 안전한 이동을 위해 경유 자세를 거칩니다.
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                
                # 목표 좌표로 이동합니다.
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

# ====================================================
# 5. 메인 로직 및 키 이벤트 처리
# ====================================================

def main():
    # --- 5-1. MyCobot 연결 및 초기화 ---
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.power_on()
        print(f"\n🤖 MyCobot 연결 성공: {PORT}. 초기 상태: 파워 ON (서보 잠금)")

        # 그리퍼 초기화 로직
        mc.set_gripper_mode(0)
        mc.init_electric_gripper()
        time.sleep(2)
        mc.set_electric_gripper(0)
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED, 1) # GRIPPER_OPEN_VALUE (85)로 열림
        time.sleep(2)
        print(f"✅ 그리퍼 초기화 완료. 위치: **{GRIPPER_OPEN_VALUE} (열림)**.")
        
    except Exception as e:
        print(f"\n❌ MyCobot 연결 실패 ({PORT}): {e}")
        sys.exit(1)

    # --- 5-2. 카메라 연결 ---
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"\n❌ 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        mc.close()
        sys.exit(1)
    
    # --- 5-3. 데이터 디렉토리 및 CSV 파일 준비 ---
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # CSV 헤더: Vision-Guided 제어에 필요한 픽셀 및 로봇 좌표 데이터 기록
            writer.writerow(['Timestamp', 'Target_Center_U', 'Target_Center_V', 'Robot_Coord_X', 'Robot_Coord_Y'])
        print(f"✅ 데이터 기록 파일 생성 완료: {CSV_FILE}")

    # 💡 Target Center 픽셀 좌표 초기화 (마지막으로 검출된 위치)
    last_center_u = None
    last_center_v = None

    print(f"✅ 현재 카메라 창 크기: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} 픽셀")
    print("\n--- 🔑 로봇 제어 가이드 ---")
    print(" [q]: 종료 | [s]: 티칭 시작(서보 해제) | [e]: 티칭 종료(서보 잠금)")
    print(" [0]: 0도 자세 | [1]: 컨베이어 캡처 자세 | [2]: 픽업 자세 (테스트)")
    print(" [3]: 로봇팔 위 캡처 자세 | [4]: Vision-Guided 픽업 | [5]: 기준 좌표 이동")
    print(" [j]: Joint 값 읽기 | [k]: Coordinates 읽기 | [g/h]: 그리퍼 닫기/열기")
    print(" [c]: 현재 화면 캡처 및 좌표 기록")
    print(f" [r]: {COORDINATE_FILE} 파일의 좌표 로드 및 이동 (추가됨)")
    print(" [w/x]: X+1mm / X-1mm 이동 | [d/a]: Y+1mm / Y-1mm 이동")
    print("----------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            # print("프레임 수신 실패.", end='\r')
            time.sleep(0.1)
            continue
        
        # --- 5-4. 비전 처리 및 시각화 ---
        
        # 1. 관심 영역(ROI) 및 목표 중심 표시
        roi_center_x, roi_center_y = (roi_start[0] + roi_end[0]) // 2, (roi_start[1] + roi_end[1]) // 2
        cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
        cv2.circle(frame, (roi_center_x, roi_center_y), 5, (0, 0, 0), -1) 
        cv2.putText(frame, "ROI / Target", (roi_center_x + 10, roi_center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 2. 빨간색 물체 검출 및 중심 표시 (초록색)
        center_x, center_y, largest_contour = find_red_center(frame)
        
        if center_x is not None:
            last_center_u, last_center_v = center_x, center_y
            
            # 외곽선 및 중심 표시
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # 검출된 객체 (초록색)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1) 
            cv2.putText(frame, f"ROI U(X): {roi_center_x}, ROI V(Y): {roi_center_y}", 
                        (roi_center_x - 310, roi_center_y + 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Detected U(X): {center_x}, Detected V(Y): {center_y}", 
                        (roi_center_x - 310, roi_center_y + 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Diff. U(X): {(roi_center_x-center_x)}, Diff. V(Y): {(roi_center_y-center_y)}", 
                        (roi_center_x - 310, roi_center_y + 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Target Not Found", (roi_center_x - 310, roi_center_y + 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 빨간색 텍스트

        cv2.imshow('MyCobot Live Camera', frame)

        # --- 5-5. 키 이벤트 처리 ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n👋 프로그램 종료 명령 수신. 자원을 해제합니다...")
            break
            
        elif key == ord('r'): # ✅ R 키: 좌표 파일 로딩 및 이동
            load_and_move_coords(mc, COORDINATE_FILE)
            
        elif key == ord('s'): # 서보 잠금 해제 (티칭 시작)
            print("\n▶️ **티칭 모드 시작** (모든 서보 잠금 해제, 수동 제어 가능)")
            mc.release_all_servos()
            
        elif key == ord('e'): # 서보 잠금 (티칭 종료)
            print("\n⏸️ **티칭 모드 종료** (모든 서보 잠금, 로봇 움직임 고정)")
            mc.power_on()

        elif key in [ord('w'), ord('x'), ord('a'), ord('d')]:
            # 1. 현재 로봇 좌표 읽기
            current_coords = mc.get_coords()
            
            # 읽기 실패 시, GLOBAL_TARGET_COORDS를 기준으로 사용 (안전 대책)
            if not isinstance(current_coords, list) or all(c == -1 for c in current_coords):
                current_coords = list(GLOBAL_TARGET_COORDS)
                print("⚠️ 로봇 좌표를 읽을 수 없어 기준 좌표를 사용합니다.")
            else:
                current_coords = list(current_coords) 
            
            # 2. 이동량 설정 (요청된 대로 1mm 단위로 다시 수정)
            move_x, move_y = 0.0, 0.0
            axis_name = ""
            
            if key == ord('w'):
                move_x = 5 # X 증가
                axis_name = "X + 1mm"
            elif key == ord('x'):
                move_x = -5 # X 감소
                axis_name = "X - 1mm"
            elif key == ord('d'): 
                move_y = 5 # Y 증가
                axis_name = "Y + 1mm"
            elif key == ord('a'): 
                move_y = -5 # Y 감소
                axis_name = "Y - 1mm"
            
            # 3. 새로운 좌표 계산 및 이동 명령 전송
            if axis_name:
                current_coords[0] += move_x
                current_coords[1] += move_y
                
                # Z, Rx, Ry, Rz 값은 유지
                mc.send_coords(current_coords, MOVEMENT_SPEED - 30) # 조금 느린 속도로 이동
                time.sleep(0.1) # 짧은 대기 시간
                
                print(f"\n➡️ 증분 이동 ({axis_name}): 새로운 좌표 (X:{current_coords[0]:.2f}, Y:{current_coords[1]:.2f})")

        elif key == ord('0'): # 0도 자세
            print(f"\n🔄 로봇을 0도 자세 이동 시작...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED) 
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            print("✅ 0도 자세 이동 완료.")
        
        elif key == ord('1'): # 컨베이어 캡처 자세
            print(f"\n🚀 컨베이어 캡처 자세 ({CONVEYOR_CAPTURE_POSE})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")
            
        elif key == ord('2'): # 테스트 픽업 자세 (관절 각도)
            print(f"\n⬇️ 테스트 픽업 자세 ({TEST_PICK_POSE})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(TMP_PICK_POSE, MOVEMENT_SPEED - 30) 
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(TEST_PICK_POSE, MOVEMENT_SPEED) 
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
            print("✅ TEST_PICK_POSE 이동 완료.")
        
        elif key == ord('3'): # 로봇팔 위 캡처 자세
            print(f"\n🚀 로봇팔 위 캡처 자세 ({ROBOTARM_CAPTURE_POSE})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")
        
        elif key == ord('4'): # Vision-Guided 정렬 및 픽업 시도 (핵심 로직)
            print(f"\n🚀 '4' 키 입력: **Vision-Guided 자동 정렬 시작**")
            
            # 1. 정렬 수행 (로봇을 계산된 목표 좌표로 이동)
            if align_to_target(mc, cap):
                # 2. 픽업 (그리퍼 닫기)
                print("✊ 그리퍼 닫는 중 (픽업 동작)...")
                mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
                time.sleep(GRIPPER_ACTION_DELAY)
                print(f"✅ 픽업(그리퍼 닫힘) 완료.")
            else:
                print("❌ 정렬에 실패하여 픽업 동작을 건너뜁니다.")

        elif key == ord('5'):
            print(f"\n🚀 GLOBAL_TARGET_COORDS ({GLOBAL_TARGET_COORDS})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_coords(GLOBAL_TARGET_COORDS, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ GLOBAL_TARGET_COORDS 이동 완료.")

        elif key == ord('c'): # 현재 카메라 프레임 캡처 및 좌표 기록 (픽셀-MM 교정용)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"
            save_path = os.path.join(DATA_DIR, filename)
            
            cv2.imwrite(save_path, frame)
            
            # 픽셀 좌표와 로봇 좌표 기록
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
                print(f"\n🔴 빨간색 물체가 검출되지 않아 캡처만 저장됨: {save_path}")

        elif key == ord('j'): # Joint 값 읽기
            current_angles = mc.get_angles()
            if isinstance(current_angles, list) and not all(c == -1 for c in current_angles): 
                print(f"\n📐 현재 Joint 값 (J1~J6): **{current_angles}**")
            else:
                print("\n❌ Joint 값을 읽을 수 없습니다. 로봇 연결 상태를 확인하세요.")

        elif key == ord('k'): # Coordinates (좌표 값) 읽기
            current_coords = mc.get_coords()
            if isinstance(current_coords, list) and not all(c == -1 for c in current_coords): 
                print(f"\n🗺️ 현재 Coordinates (X, Y, Z, R, P, Y): **{current_coords}**") 
            else:
                print("\n❌ Coordinates 값을 읽을 수 없습니다. 로봇 연결 상태를 확인하세요.")
        
        elif key == ord('g'): # 그리퍼 닫기
            print("\n✊ 그리퍼 닫는 중...")
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED) 
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 닫힘 완료 (위치: **{GRIPPER_CLOSED_VALUE}**).")
            
        elif key == ord('h'): # 그리퍼 열기
            print("\n👐 그리퍼 여는 중...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
            time.sleep(GRIPPER_ACTION_DELAY)
            print(f"✅ 그리퍼 열림 완료 (위치: **{GRIPPER_OPEN_VALUE}**).")

    # --- 5-6. 종료 시 자원 해제 ---
    print("🧹 자원 해제 중: 카메라 및 로봇 연결 종료...")
    cap.release()
    cv2.destroyAllWindows()
    try:
        mc.close()
    except Exception:
        pass
    print("👍 프로그램 종료 완료.")

if __name__ == "__main__":
    main()