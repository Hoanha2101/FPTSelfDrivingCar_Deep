DIRECTION_LEFT = "X"
DIRECTION_RIGHT = "Y"
DIRECTION_STRAIGHT = "S"

# Phần trăm mặt đường sẽ lấy
per_len_lane = 0.9

# Ngưỡng quay bánh lại
back_threshold = 5

# ngưỡng lệch góc thì phải push ngay
threshold_scale = 5 

# Ngưỡng thu report 
count_control = 60

# Các điểm liên quan đến xe (điểm trụ sở, padding từ 2 bên)
car_length_padding = 100

# Setting TensorRT
input_names = ['images']
output_names = ['output']
batch = 1
plan = "models/tusimple_18_FP16.trt"