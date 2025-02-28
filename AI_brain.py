import cv2
import math
import numpy as np
from utils_func import CLEAN_DATA_CSV_DIRECTION, ADD_DATA_CSV_MASK_DIRECTION, ADD_DATA_CSV_DIRECTION_STRAIGHT, CLEAN_DATA_CSV_DIRECTION_STRAIGHT,CHECK_PUSH, csv_path, csv_mask_path, csv_straight_path, csv_back_control_path, CLEAN_DATA_CSV_BACK_CONTROL
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType, ModelConfig
import pandas as pd
from setting_AI import *

def draw_direction_arrow(img, center, angle_deg, size=50, color=(0, 255, 255)):
    """
    Vẽ biểu tượng mũi tên chỉ hướng xoay theo góc angle_deg tại vị trí center.
    Mũi tên mặc định chỉ lên trên, khi quay theo góc, biểu tượng sẽ phản ánh hướng lái.
    """
    # Định nghĩa các điểm của mũi tên (mặc định hướng lên trên)
    pts = np.array([
        [0, -size],               # điểm mũi tên (đỉnh)
        [-size // 4, size // 2],    # góc trái dưới
        [0, size // 4],           # điểm giữa dưới
        [size // 4, size // 2]      # góc phải dưới
    ], dtype=np.float32)
    
    # Tạo ma trận xoay
    M = cv2.getRotationMatrix2D((0, 0), angle_deg, 1)
    rotated_pts = np.dot(pts, M[:, :2])
    # Dịch các điểm về vị trí center
    rotated_pts[:, 0] += center[0]
    rotated_pts[:, 1] += center[1]
    rotated_pts = rotated_pts.astype(np.int32)
    
    cv2.fillPoly(img, [rotated_pts], color)

# ---------------------------- CONFIG -----------------------------------------


# Cấu hình model và video
model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True

# Khởi tạo model phát hiện làn đường
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
cfg_ = ModelConfig(model_type)
height = cfg_.img_h
width = cfg_.img_w


car_point_left  = (car_length_padding, height)
car_point_right = (width - car_length_padding, height)
car_center_bottom = ((car_point_left[0] + car_point_right[0]) // 2, height)
car_center_top = (car_center_bottom[0], 0)

# -------------------------------------------------------------------------------

CLEAN_DATA_CSV_DIRECTION()
CLEAN_DATA_CSV_DIRECTION_STRAIGHT()
CLEAN_DATA_CSV_BACK_CONTROL()

dr_back_control = None
an_back_control = None
len_csv_control_back = None

def AI(frame, paint = False, resize_img = True):
    global dr_back_control, an_back_control, len_csv_control_back
    PUSH_RETURN = None
    visualization_img, lane_left_top, lane_right_top, lane_left_bottom, lane_right_bottom = lane_detector.detect_lanes(frame)

    if paint:
        # Vẽ các điểm đánh dấu xe
        cv2.circle(visualization_img, car_point_left, 10, (50, 100, 255), -1)
        cv2.circle(visualization_img, car_center_bottom, 10, (50, 100, 255), -1)
        cv2.circle(visualization_img, car_point_right, 10, (50, 100, 255), -1)
        cv2.circle(visualization_img, car_center_top, 10, (50, 100, 255), -1)
    
    if lane_left_top is not None and lane_right_top is not None:
        # Tính điểm top center của làn đường
        top_center = ((lane_left_top[0] + lane_right_top[0]) // 2,
                      (lane_left_top[1] + lane_right_top[1]) // 2)
        if paint:
            # Vẽ các điểm top lane và top center
            cv2.circle(visualization_img, lane_left_top, 5, (0, 255, 255), -1)
            cv2.circle(visualization_img, lane_right_top, 5, (0, 255, 255), -1)
            cv2.circle(visualization_img, top_center, 7, (0, 0, 255), -1)
        
        # Vẽ điểm kiểm soát ở chân của lane top
        point_control_left  = (lane_left_top[0], height)
        point_control_right = (lane_right_top[0], height)
        
        if paint:
            cv2.circle(visualization_img, point_control_left, 10, (100, 255, 100), -1)
            cv2.circle(visualization_img, point_control_right, 10, (100, 255, 100), -1)
        
        # Tính góc lái dựa trên vector nối giữa car_center_bottom và top_center
        dx = top_center[0] - car_center_bottom[0]
        dy = car_center_bottom[1] - top_center[1]
        angle_rad = math.atan2(dx, dy)
        angle_deg = angle_rad * 180 / math.pi
        
        # Xác định hướng lái với ngưỡng ±5 độ
        threshold = 5
        if angle_deg < -threshold:
            direction = DIRECTION_LEFT
            
        elif angle_deg > threshold:
            direction = DIRECTION_RIGHT
            
        else:
            direction = DIRECTION_STRAIGHT
        
        if paint:   
            # Hiển thị thông tin văn bản: hướng lái và góc
            text = f"{direction} ({angle_deg:.2f} deg)"
            cv2.rectangle(visualization_img, (10, 10), (460, 70), (0, 0, 0), -1)  # Nền cho text (để dễ đọc)
            cv2.putText(visualization_img, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            icon_center = (width - 80, 80)  
            draw_direction_arrow(visualization_img, icon_center, angle_deg, size=40, color=(0, 200, 200))
            cv2.circle(visualization_img, icon_center, 45, (0, 200, 200), 2)
        
        if direction != DIRECTION_STRAIGHT:
            ADD_DATA_CSV_MASK_DIRECTION(direction, abs(int(angle_deg)))
        else: 
            ADD_DATA_CSV_DIRECTION_STRAIGHT(direction, abs(int(angle_deg)))
            
        push, dr_back, an_back = CHECK_PUSH()
        
        if push is not None:
            dr_back_control = dr_back
            an_back_control = an_back
            
            df_csv_ = pd.read_csv(csv_straight_path)
            len_csv_control_back = len(df_csv_)
            
            print("Push_next:",push)
            
            PUSH_RETURN = push
        
        if dr_back_control is not None and an_back_control is not None:
            df_csv_ = pd.read_csv(csv_straight_path)
            if len(df_csv_) - len_csv_control_back >= back_threshold:
                df_back_control_csv = pd.read_csv(csv_back_control_path)
                an_back_control = sum(df_back_control_csv["angle"])

                if dr_back_control == DIRECTION_RIGHT:
                    push_back = f"{DIRECTION_LEFT}:{an_back_control:03d}"
                else:
                    push_back = f"{DIRECTION_RIGHT}:{an_back_control:03d}"
                    
                print("Push_back:", push_back)
                PUSH_RETURN = push_back
                dr_back_control = None
                an_back_control = None
                len_csv_control_back = None
                CLEAN_DATA_CSV_BACK_CONTROL()
    
    if resize_img:    
        visualization_img = cv2.resize(visualization_img, (visualization_img.shape[1] // 2, visualization_img.shape[0] // 2))
    
    return visualization_img, PUSH_RETURN