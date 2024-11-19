
import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np


# Khởi tạo thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Khởi tạo MTCNN
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# Thư mục chứa dữ liệu gốc
data_dir = 'dataset'

# Thư mục để lưu ảnh khuôn mặt đã được căn chỉnh
aligned_data_dir = 'aligned_dataset'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(aligned_data_dir, exist_ok=True)

# Duyệt qua các lớp (tên người)
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    aligned_class_dir = os.path.join(aligned_data_dir, class_name)
    os.makedirs(aligned_class_dir, exist_ok=True)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        aligned_img_path = os.path.join(aligned_class_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        # Sử dụng MTCNN để phát hiện và căn chỉnh khuôn mặt
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            # Chuyển đổi tensor sang NumPy array
            img_cropped = img_cropped.permute(1, 2, 0).numpy()
            # Nhân với 255 và chuyển đổi sang uint8
            img_cropped = (img_cropped * 255).astype(np.uint8)
            # Chuyển từ RGB sang BGR để OpenCV xử lý đúng màu
            img_cropped = Image.fromarray(img_cropped, 'RGB')
            # Lưu ảnh đã căn chỉnh
            img_cropped.save(aligned_img_path)
            print(f"Đã lưu ảnh căn chỉnh: {aligned_img_path}")
        else:
            print(f"Không phát hiện khuôn mặt trong ảnh: {img_path}")
