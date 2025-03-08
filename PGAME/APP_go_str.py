import pygame
import cv2
import time
import serial
import sys
from PGAME.AI_brain_TRT_go_str import AI_TRT
from setting_AI import *
from PGAME.utils_func_go_str import CLEAN_DATA_CSV_DIRECTION, ADD_DATA_CSV_MASK_DIRECTION, ADD_DATA_CSV_DIRECTION_STRAIGHT, CLEAN_DATA_CSV_DIRECTION_STRAIGHT, CHECK_PUSH, CLEAN_DATA_CSV_BACK_CONTROL

serial_port = serial.Serial(
    port="COM8",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)


# Initialize pygame
pygame.init()

# Screen settings
screen_width = 1536
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("AI Camera Control")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Fonts
font = pygame.font.Font(None, 50)
small_font = pygame.font.Font(None, 36)  # Font nhỏ hơn để hiển thị kết quả push

# Buttons
start_button = pygame.Rect(100, 750, 200, 50)
end_button = pygame.Rect(400, 750, 200, 50)

# Initialize camera
# cap = cv2.VideoCapture("videos/g.mp4")

cap =cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 

# Wait for serial port to initialize
time.sleep(1)

time_stop = sys.maxsize
sleep_time = sys.maxsize
running = True
active = False  # Flag to track whether the AI processing is active
clear = True
push_results = []  # List lưu 5 kết quả push gần nhất

while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_button.collidepoint(event.pos):
                active = True
                clear = True
            elif end_button.collidepoint(event.pos):
                active = False
                print("Stopped pushing")

    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    visualization_img = frame
    
    if active:
        if clear:
            CLEAN_DATA_CSV_DIRECTION()
            CLEAN_DATA_CSV_DIRECTION_STRAIGHT()
            # CLEAN_DATA_CSV_BACK_CONTROL()
            clear = False
        visualization_img, PUSH_RETURN = AI_TRT(frame, paint=True, resize_img=True)
       
        if PUSH_RETURN:
            serial_port.write("x:000".encode())
            serial_port.write(PUSH_RETURN.encode())

            push_results.append(PUSH_RETURN)  # Thêm kết quả mới vào danh sách
            if len(push_results) > 5:
                push_results.pop(0)  # Xóa kết quả cũ nhất nếu danh sách dài hơn 5

            angle = int(PUSH_RETURN.split(":")[1])
            angle = min(30, angle)
            sleep_time = angle / ROTATION_SPEED
            time_stop = time.time()
        
        if time.time() - time_stop >= sleep_time:
            serial_port.write("x:000".encode())

            push_results.append("x:000")
            
            if len(push_results) > 5:
                push_results.pop(0)
            
            time_stop = sys.maxsize
    
    pygame_frame = pygame.surfarray.make_surface(cv2.rotate(cv2.flip(visualization_img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE))
    screen.blit(pygame_frame, (10, 10))

    pygame.draw.rect(screen, GREEN if active else BLACK, start_button)
    pygame.draw.rect(screen, RED, end_button)
    
    start_text = font.render("Start", True, WHITE)
    end_text = font.render("End", True, WHITE)
    screen.blit(start_text, (start_button.x + 60, start_button.y + 10))
    screen.blit(end_text, (end_button.x + 70, end_button.y + 10))

    # Hiển thị kết quả push gần nhất
    for i, result in enumerate(reversed(push_results)):
        result_text = small_font.render(result, True, BLACK)
        screen.blit(result_text, (1400, 600 - i * 40))  # Hiển thị từ dưới lên trên
    
    pygame.display.flip()

cap.release()
cv2.destroyAllWindows()
pygame.quit()

