import time
# pynput 같은 라이브러리를 사용하면 Enter 없이 키 입력 가능 (별도 설치 필요)
# 하지만 간단한 테스트를 위해 표준 input()을 사용합니다.
from pymycobot import MyCobot320

# --- 상수 정의 ---
TEST_PICK_POSE_WIDTH = [-237.90, 20, 183.6, -174.98, 0, 0]
MOVEMENT_SPEED = 70
SEQUENTIAL_MOVE_DELAY = 1.5
GRIPPER_SPEED = 50
GRIPPER_ACTION_DELAY = 1

ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90]

GRIPPER_OPEN_VALUE = 85
GRIPPER_CLOSED_VALUE = 25

# --- 로봇 연결 ---
PORT = "COM3"
BAUD = 115200

try:
    mc = MyCobot320(PORT, BAUD)
    print(f"✅ MyCobot320 연결 성공: {PORT} @ {BAUD}")
except Exception as e:
    print(f"❌ MyCobot320 연결 실패! 포트와 전원을 확인하세요. 에러: {e}")
    # 연결 실패 시 프로그램 종료
    exit()

# --- 초기 상태 설정 ---
# 로봇의 현재 상태를 저장할 변수 (주의: 이 변수는 로봇의 실제 좌표를 항상 반영하지는 않음)
target_pose = list(TEST_PICK_POSE_WIDTH) 
# list()를 사용하여 초기 값 복사

# 초기화: 안전 포즈로 이동
safe_pose = list(target_pose)
safe_pose[2] += 50
print("🚀 초기화: 안전 포즈로 이동 중...")
mc.send_coords(safe_pose, MOVEMENT_SPEED)
time.sleep(SEQUENTIAL_MOVE_DELAY * 2) # 초기 동작은 여유 있게 대기

# 그리퍼 초기화: 열기
print("🤚 그리퍼 열기")
mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
time.sleep(GRIPPER_ACTION_DELAY)

# --- 메인 루프 (키 입력 처리) ---
print("\n--- 로봇 테스트 시작 ---")
print("1: 물건 잡기 동작 시작 (Z-이동 -> 하강 -> 그리퍼 닫기)")
print("2: 물건 들고 상승")
print("3: Capture Pose로 이동 (Z: 10)")
print("q: 프로그램 종료")
print("--------------------------")

while True:
    try:
        key_input = input("키를 입력하세요 (1, 2, 3, q): ").strip().lower()
    except EOFError:
        # 콘솔이 닫히면 종료
        break
    
    # Q: 프로그램 종료
    if key_input == 'q':
        print("\n👋 프로그램을 종료합니다.")
        break

    # 1: 물건 잡기 동작
    elif key_input == '1':
        print("\n--- 1. 하강 및 잡기 시작 ---")
        
        # 1-1: 물건 위치로 하강
        print(f"1-1: X:{target_pose[0]:.2f}, Y:{target_pose[1]:.2f} (Z:{target_pose[2]:.2f}) 로 하강.")
        # target_pose는 TEST_PICK_POSE_WIDTH와 동일
        mc.send_coords(target_pose, MOVEMENT_SPEED - 30) # 느린 속도
        time.sleep(SEQUENTIAL_MOVE_DELAY)
        
        # 1-2: 그리퍼 닫기
        print(f"1-2: 그리퍼 닫기 (Value: {GRIPPER_CLOSED_VALUE})")
        mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
        time.sleep(GRIPPER_ACTION_DELAY)

    # 2: 물건 들고 상승
    elif key_input == '2':
        print("\n--- 2. 상승 시작 ---")
        
        # target_pose의 Z 좌표를 100 증가
        target_pose[2] += 100
        
        # 2-1: 상승된 위치로 이동
        print(f"2-1: X:{target_pose[0]:.2f}, Y:{target_pose[1]:.2f} (Z:{target_pose[2]:.2f}) 로 상승.")
        mc.send_coords(target_pose, MOVEMENT_SPEED) # 일반 속도
        time.sleep(SEQUENTIAL_MOVE_DELAY)

    # 3: Capture Pose로 이동
    elif key_input == '3':
        print("\n--- 3. Capture Pose로 이동 시작 ---")
        
        # 3-1: Capture Pose로 이동
        print(f"3-1: Capture Pose (X:{ROBOTARM_CAPTURE_POSE[0]:.2f}, Y:{ROBOTARM_CAPTURE_POSE[1]:.2f}, Z:{ROBOTARM_CAPTURE_POSE[2]:.2f}) 로 이동.")
        mc.send_coords(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
        time.sleep(SEQUENTIAL_MOVE_DELAY)
        
        # 3-2: Capture Pose 도착 후 그리퍼 열기 (다음 동작을 위해)
        print(f"3-2: 그리퍼 열기 (Value: {GRIPPER_OPEN_VALUE})")
        mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED)
        time.sleep(GRIPPER_ACTION_DELAY)
        
        # target_pose를 Capture Pose로 업데이트 (다음 1번 동작을 대비하여)
        target_pose = list(ROBOTARM_CAPTURE_POSE)

    else:
        print(f"🚨 알 수 없는 입력입니다: {key_input}. '1', '2', '3', 'q' 중 하나를 입력하세요.")