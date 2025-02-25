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


# Cấu hình model và video
model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = True

# Khởi tạo model phát hiện làn đường
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def convert_pytorch2onnx(model, input_samples, path_onnx, mode='float32bit', device='cuda'):
    if mode == 'float16bit':
        print("Converting model and inputs to float16")
        model = model.half()  # Convert model to float16
        input_samples = input_samples.half()  # Convert input samples to float16
    elif mode == 'float32bit':
        print("Converting model and inputs to float32")
        model = model.float()  # Convert model to float32
        input_samples = input_samples.float()  # Convert input samples to float32
    
    model.to(device)
    model.eval()
    input_samples = input_samples.to(device)
    
    torch.onnx.export(
        model,  # The model
        input_samples,  # Input tensor with desired size
        path_onnx,  # Path to save the ONNX file
        verbose=False,  # Whether to print the process
        opset_version=12,  # ONNX opset version
        do_constant_folding=True,  # Whether to do constant folding optimization
        input_names=['images'],  # Model input names
        output_names=['output'],  # Model output names
        dynamic_axes={
            'images': {0: 'batch_size'},  # Dynamic batch size for inputs
            'output': {0: 'batch_size'}  # Dynamic batch size for outputs
        }
    )


input_samples = torch.randn(1, 3, 288, 800)  # Example input tensor
path_onnx = "models/tusimple_18.onnx"
# Convert the model to ONNX
convert_pytorch2onnx(lane_detector.model, input_samples, path_onnx, mode='float32bit', device=device)
