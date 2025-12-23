import time
import sys
from pymycobot import MyCobot320

# ===============================================
# ⚙️ 로봇 설정 및 좌표 정의
# ===============================================
PORT = "COM3"
BAUD = 115200
MOVEMENT_SPEED = 10
DELAY = 2.0  # 이동 후 대기 시간

# 각도 기반 포즈 (Joint Angles)
CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90]
ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90]
INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86]
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 좌표 기반 포즈 (Coordinates: X, Y, Z, RX, RY, RZ)
BASE_PICK_COORDS = [-240, 20, 180, -174.98, 0, 0]
TMP_BASE_PICK_COORDS = [-240, 20, 250, -174.98, 0, 0]

def test_robot():
    try:
        mc = MyCobot320(PORT, BAUD)
        mc.power_on()
        print(f"✅ MyCobot 연결 성공: {PORT}")
        
        print("\n--- 🔑 테스트 키 가이드 ---")
        print(" [3]: 컨베이어 캡처 포즈 이동")
        print(" [2]: 로봇팔 위 캡처 포즈 이동")
        print(" [1]: 중간 대기 포즈 이동")
        print(" [0]: 제로 포즈 이동")
        print(" [p]: 픽업 좌표 시퀀스 테스트 (접근 -> 하강)")
        print(" [q]: 종료")
        print("----------------------------")

        while True:
            key = input("\n명령을 입력하세요: ").lower()

            if key == '3':
                print(f"🚀 컨베이어 캡처 포즈로 이동: {CONVEYOR_CAPTURE_POSE}")
                mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
                
            elif key == '2':
                print(f"🚀 로봇팔 위 캡처 포즈로 이동: {ROBOTARM_CAPTURE_POSE}")
                mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
                
            elif key == '1':
                print(f"🚀 중간 대기 포즈로 이동: {INTERMEDIATE_POSE_ANGLES}")
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                
            elif key == '0':
                print(f"🚀 제로 포즈로 이동: {ZERO_POSE_ANGLES}")
                mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
                
            elif key == 'p':
                print(f"⬇️ 픽업 지점 하강: {BASE_PICK_COORDS}")
                mc.send_coords(BASE_PICK_COORDS, MOVEMENT_SPEED - 5) # 하강 시 조금 더 천천히
            
            elif key == 'o':
                print(f"임시 픽업 지점 하강: {TMP_BASE_PICK_COORDS}")
                mc.send_coords(TMP_BASE_PICK_COORDS, MOVEMENT_SPEED - 5) # 하강 시 조금 더 천천히
                
            elif key == 'q':
                print("👋 테스트를 종료합니다.")
                break
            
            else:
                print("⚠️ 정의되지 않은 키입니다.")

            time.sleep(0.1)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        mc.power_off() # 안전을 위해 전원 해제 (필요에 따라 유지 가능)

if __name__ == "__main__":
    test_robot()