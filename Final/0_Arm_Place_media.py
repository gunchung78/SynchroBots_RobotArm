import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# --- 설정값 ---
CAMERA_INDEX = 0
CLASS_NAMES = ["ESP32", "L298N", "MB102"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def round_coords(coords, precision=2):
    if isinstance(coords, (list, tuple, np.ndarray)):
        return [round(float(c), precision) for c in coords]
    return round(float(coords), precision)

def get_vision_rz_streaming(frame):
    """
    ROI 영역 내에서 물체를 검출하고, 실시간으로 스트리밍 화면을 보여줍니다.
    """
    # 1. 관심 영역(ROI) 설정 (원래 코드의 범위)
    roi = frame[70:330, 90:390].copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 2. 물체 마스크 생성 (흰색 배경 위 물체 추출용)
    # 배경 상황에 따라 lower/upper 값을 조정하세요.
    mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_rz = 0
    
    if contours:
        # 가장 큰 물체 선택
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 500: # 노이즈 제거
            # 최소 면적 사각형 (기울어진 박스) 계산
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            # --- 디버깅 시각화 ---
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(roi, [box], 0, (0, 255, 0), 2) # 초록색 박스
            
            # 각도 계산 (MyCobot J6/Rz 축에 맞게 보정 필요)
            final_rz = round_coords(angle, 2)
            
            # 정보 표시
            cv2.putText(roi, f"Angle: {final_rz}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 스트리밍 창 표시
    cv2.imshow("Vision Debug - ROI", roi)
    cv2.imshow("Mask Debug", mask)
    
    return final_rz

def debug_pick_vision():
    """
    메인 루프: 카메라를 열고 실시간으로 Pick 각도를 추론하며 화면을 보여줍니다.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("--- 실시간 비전 디버깅 시작 (종료하려면 'q'를 누르세요) ---")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 각도 추론 및 화면 표시
        rz_angle = get_vision_rz_streaming(frame)
        
        # 2. MyCobot에 적용될 최종 Pick 각도 로직 미리보기
        # (제시해주신 보정 로직 적용)
        display_rz = rz_angle
        if display_rz < 0:
            display_rz = abs(display_rz)
        elif 1 < display_rz < 44:
            display_rz = 90 - display_rz
        elif 46 < display_rz < 90:
            display_rz = abs(90 - display_rz)
            
        # 콘솔에 실시간 출력
        print(f"\rRaw Angle: {rz_angle:6.2f} | Final Pick Rz: {display_rz:6.2f}", end="")

        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_pick_vision()