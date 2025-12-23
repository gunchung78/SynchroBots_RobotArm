import cv2
import sys

def main():
    cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        cv2.imshow('MyCobot Live Camera', frame)

        # q 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
