import cv2
from AI_brain import AI
from AI_brain_TRT import AI_TRT
import time
import serial
import sys

cap =cv2.VideoCapture(0, cv2.CAP_DSHOW)

serial_port = serial.Serial(
    port="COM5",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

# Wait a second to let the port initialize
time.sleep(1)

ROTATION_SPEED = 10  # Tốc độ quay:  độ/s
time_stop = sleep_time = sys.maxsize

while True:
    _,frame = cap.read()
    visualization_img, PUSH_RETURN = AI_TRT(frame, paint = True, resize_img = False)
 
    if PUSH_RETURN:
        serial_port.write("x:000".encode())
        time.sleep(0.5) 
        serial_port.write(PUSH_RETURN.encode())
        print(PUSH_RETURN)
        angle = int(PUSH_RETURN.split(":")[1])
        sleep_time = angle / ROTATION_SPEED
        time_stop = time.time()
    
    if time.time() - time_stop >= sleep_time:
        time.sleep(0.5) #  ------------------
        print()
        print("x:000")
        serial_port.write("x:000".encode())
        print()
        time_stop = sys.maxsize
        
    cv2.imshow("Detected lanes", visualization_img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    time.sleep(0.1)