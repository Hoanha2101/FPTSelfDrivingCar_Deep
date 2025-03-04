import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)



if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận dữ liệu từ camera")
        break

    fps = 1.0 / (time.time() - start_time)  # Tính FPS
    print(f"FPS: {fps:.2f}")

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
