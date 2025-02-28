import cv2
from AI_brain import AI
from AI_brain_TRT import AI_TRT
import time

cap = cv2.VideoCapture("videos/test1.mp4")

while cap.isOpened():
    
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # visualization_img, PUSH_RETURN = AI(frame, paint = True, resize_img = True)
    visualization_img, PUSH_RETURN = AI_TRT(frame, paint = True, resize_img = True)
    
    if PUSH_RETURN:
        print("-----",PUSH_RETURN)
        
    # Hiển thị FPS lên hình
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    # print(f"FPS: {fps:.2f}")
    
    cv2.imshow("Detected lanes", visualization_img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()