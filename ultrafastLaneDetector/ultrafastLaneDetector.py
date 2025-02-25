import sys
sys.path.append("./.")
import cv2
import torch
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist


from ultrafastLaneDetector.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

		self.use_gpu = use_gpu

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path, self.cfg, use_gpu)

		# Initialize image transformation
		self.img_transform = self.initialize_image_transform()

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Load the model architecture
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
						use_aux=False) # we dont need auxiliary segmentation in testing


		# Load the weights from the downloaded model
		if use_gpu:
			if torch.backends.mps.is_built():
				net = net.to("mps")
				state_dict = torch.load(model_path, map_location='mps')['model'] # Apple GPU
			else:
				net = net.cuda()
				state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
		else:
			state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

		compatible_state_dict = {}
		for k, v in state_dict.items():
			if 'module.' in k:
				compatible_state_dict[k[7:]] = v
			else:
				compatible_state_dict[k] = v

		# Load the weights into the model
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()

		return net

	@staticmethod
	def initialize_image_transform():
		# Create transfom operation to resize and normalize the input images
		img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		return img_transforms

	def detect_lanes(self, image, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

		# Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

		return visualization_img

	def prepare_input(self, img):
		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(img)
		input_img = self.img_transform(img_pil)
		input_tensor = input_img[None, ...]

		if self.use_gpu:
			if not torch.backends.mps.is_built():
				input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):
		with torch.no_grad():
			# print("input: ", input_tensor.shape)
			output = self.model(input_tensor)
			# print("output: ", output.shape)
		return output

	@staticmethod
	def process_output(output, cfg):		
		# Parse the output of the model
		processed_output = output[0].data.cpu().numpy()
		processed_output = processed_output[:, ::-1, :]
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc


		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
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
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return np.array(lanes_points, dtype=object), np.array(lanes_detected, dtype=object)

	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		left_top = None
		right_top = None
		left_bottom = None
		right_bottom = None

		# Resize ảnh đầu vào
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)

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
			y_threshold = y_lane_bottom - 0.9 * lane_length
			
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


	







