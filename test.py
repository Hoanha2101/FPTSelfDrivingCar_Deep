import cv2
from AI_brain import AI
from AI_brain_TRT import AI_TRT
import time
import sys


cap = cv2.VideoCapture("videos/f.avi")

# cap =cv2.VideoCapture(1, cv2.CAP_DSHOW)


ROTATION_SPEED = 10  # Tốc độ quay:  độ/s
time_stop = sleep_time = sys.maxsize


while cap.isOpened():
    
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # visualization_img, PUSH_RETURN = AI(frame, paint = True, resize_img = True)
    visualization_img, PUSH_RETURN = AI_TRT(frame, paint = True, resize_img = True)
    
    if PUSH_RETURN:
        print("-----","x:000")
        time.sleep(0.5) 
        print("-----",PUSH_RETURN)
        angle = int(PUSH_RETURN.split(":")[1])
        sleep_time = angle / ROTATION_SPEED
        time_stop = time.time()
    
    if time.time() - time_stop >= sleep_time:
        time.sleep(0.5) #  ------------------
        print()
        print("-----","x:000")
        print()
        time_stop = sys.maxsize
        
    # Hiển thị FPS lên hình
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    # print(f"FPS: {fps:.2f}")
    
    cv2.imshow("Detected lanes", visualization_img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()