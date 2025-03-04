import cv2
# from AI_brain import AI
# from AI_brain_TRT import AI_TRT
import time
import serial

# cap = cv2.VideoCapture(1)

serial_port = serial.Serial(
    port="COM5",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

if not serial_port.is_open:
    serial_port.open()


# Wait a second to let the port initialize
time.sleep(1)
while True:
    PUSH_RETURN = "Y:030"
    print(PUSH_RETURN)
    # serial_port.write(PUSH_RETURN.encode())
    # serial_port.write((PUSH_RETURN + "\r\n").encode())
    
    bytes_written = serial_port.write(PUSH_RETURN.encode())
    print(f"Bytes sent: {bytes_written}")


    time.sleep(2)
    
