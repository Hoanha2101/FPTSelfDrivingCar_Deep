import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from __init__ import TensorrtBase
import cv2
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import scipy.special
from setting_AI import *
from utils_func import CLEAN_DATA_CSV_DIRECTION, ADD_DATA_CSV_MASK_DIRECTION, ADD_DATA_CSV_DIRECTION_STRAIGHT, CLEAN_DATA_CSV_DIRECTION_STRAIGHT,CHECK_PUSH, csv_path, csv_mask_path, csv_straight_path, csv_back_control_path, CLEAN_DATA_CSV_BACK_CONTROL
import pandas as pd
import math

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

net = TensorrtBase(plan,
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch,
                   )

images = np.random.rand(1, 288, 800, 3).astype(np.float32)

binding_shape_map = {
    "tensor": images.shape,
    }

def INFER_TRT(images):
    # images = np.expand_dims(images, axis=0)
    images = np.ascontiguousarray(images).astype(np.float32)
    net.cuda_ctx.push()
    inputs, outputs, bindings, stream = net.buffers
    # Set optimization profile and input shape
    net.context.set_optimization_profile_async(0, stream.handle)
    net.context.set_input_shape(input_names[0], images.shape)
    
    # Transfer input data to the GPU
    cuda.memcpy_htod_async(inputs[0].device, images, stream)
    # Execute inference
    net.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)      
    # Transfer predictions back to the host
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    stream.synchronize()
    
    # Copy outputs
    trt_outputs = [out.host.copy() for out in outputs]
    net.cuda_ctx.pop()
    return trt_outputs[0].reshape(1, 101, 56, 4)

img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])
    
def prepare_input(img):
    # Transform the image for inference
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_img = img_transforms(img_pil)
    input_tensor = input_img[None, ...]

    return input_tensor

def process_output(output):		
    # Parse the output of the model
    processed_output = np.array(output[0].data)
    processed_output = processed_output[:, ::-1, :]
    prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
    idx = np.arange(100) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    processed_output = np.argmax(processed_output, axis=0)
    loc[processed_output == 100] = 0
    processed_output = loc


    col_sample = np.linspace(0, 800 - 1, 100)
    col_sample_w = col_sample[1] - col_sample[0]

    lanes_points = []
    lanes_detected = []

    max_lanes = processed_output.shape[1]
    for lane_num in range(max_lanes):
        lane_points = []
        # Check if there are any points detected in the lane
        if np.sum(processed_output[:, lane_num] != 0) > 2:

            lanes_detected.append(True)

            # Process each of the points for each lane
            for point_num in range(processed_output.shape[0]):
                if processed_output[point_num, lane_num] > 0:
                    lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * 1280 / 800) - 1, int(720 * (tusimple_row_anchor[56-1-point_num]/288)) - 1 ]
                    lane_points.append(lane_point)
        else:
            lanes_detected.append(False)

        lanes_points.append(lane_points)
    return np.array(lanes_points, dtype=object), np.array(lanes_detected, dtype=object)

def draw_lanes(input_img, lanes_points, lanes_detected, draw_points=True):
		left_top = None
		right_top = None
		left_bottom = None
		right_bottom = None

		# Resize ảnh đầu vào
		visualization_img = cv2.resize(input_img, (1280, 720), interpolation=cv2.INTER_AREA)

		# Kiểm tra nếu cả 2 lane (trái và phải) được phát hiện
		if lanes_detected[1] and lanes_detected[2]:
			lane_segment_img = visualization_img.copy()
			
			# Chuyển các điểm của lane trái và phải sang numpy array
			left_lane = np.array(lanes_points[1])
			right_lane = np.array(lanes_points[2])
			
			# Tính y_top và y_bottom của từng lane
			y_top_left = np.min(left_lane[:, 1])
			y_bottom_left = np.max(left_lane[:, 1])
			y_top_right = np.min(right_lane[:, 1])
			y_bottom_right = np.max(right_lane[:, 1])
			
			# Xác định vùng giao nhau của 2 lane theo trục y
			y_lane_top = max(y_top_left, y_top_right)
			y_lane_bottom = min(y_bottom_left, y_bottom_right)
			lane_length = y_lane_bottom - y_lane_top
			
			# Xác định ngưỡng y cho 90% chiều dài (phần gần camera)
			y_threshold = y_lane_bottom - per_len_lane * lane_length
			
			# Lọc các điểm của lane theo ngưỡng y (chỉ lấy phần gần camera)
			left_points_90 = [point for point in lanes_points[1] if point[1] >= y_threshold]
			right_points_90 = [point for point in lanes_points[2] if point[1] >= y_threshold]
			# Tính tọa độ của cạnh trên và cạnh dưới cho lane trái
			if left_points_90:
				left_top = min(left_points_90, key=lambda p: p[1])    # Điểm có y nhỏ nhất
				left_bottom = max(left_points_90, key=lambda p: p[1])   # Điểm có y lớn nhất


			# Tính tọa độ của cạnh trên và cạnh dưới cho lane phải
			if right_points_90:
				right_top = min(right_points_90, key=lambda p: p[1])
				right_bottom = max(right_points_90, key=lambda p: p[1])

			# Nếu có đủ điểm từ cả hai lane, tiến hành vẽ
			if len(left_points_90) > 0 and len(right_points_90) > 0:
				pts = np.vstack((np.array(left_points_90), np.flipud(np.array(right_points_90))))
				cv2.fillPoly(lane_segment_img, pts=[pts], color=(255,191,0))
				visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
		
		# (Nếu cần) Vẽ các điểm của lane lên ảnh
		if draw_points:
			for lane_num, lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)

		return visualization_img, left_top, right_top, left_bottom, right_bottom

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

height = 720
width = 1280

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

def AI_TRT(frame, paint = False, resize_img = True):
    global dr_back_control, an_back_control, len_csv_control_back
    PUSH_RETURN = None
    
    frame_ = prepare_input(frame)
    frame_ = INFER_TRT(frame_)
    lanes_points, lanes_detected = process_output(frame_)
    
    visualization_img, lane_left_top, lane_right_top, lane_left_bottom, lane_right_bottom = draw_lanes(frame, lanes_points, lanes_detected, draw_points=True)
    
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
