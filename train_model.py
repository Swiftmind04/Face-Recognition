
import os
import torch
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
import numpy as np

# Thiết lập tham số
data_dir = 'aligned_dataset'  # Sử dụng thư mục ảnh đã căn chỉnh
batch_size = 32

# Kiểm tra thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Tải mô hình FaceNet đã được huấn luyện sẵn
model = InceptionResnetV1(pretrained='vggface2').to(device)
model.eval()

# Chuẩn bị phép biến đổi
transformation = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Tạo dataset
dataset = datasets.ImageFolder(data_dir, transform=transformation)

# Tạo DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Trích xuất embeddings và nhãn
embeddings = []
labels = []

with torch.no_grad():
    for imgs, lbls in dataloader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        embeddings.append(outputs.cpu())
        labels.extend(lbls.numpy())

embeddings = torch.cat(embeddings).numpy()
labels = np.array(labels)

# Lưu danh sách tên lớp
class_names = dataset.classes
np.save('class_names.npy', class_names)
print("Đã lưu danh sách tên lớp.")

# Tính toán embeddings trung bình cho mỗi lớp
print("Tính toán embeddings trung bình cho mỗi lớp...")
class_embeddings = {}
for idx, class_name in enumerate(class_names):
    class_indices = np.where(labels == idx)[0]
    class_embs = embeddings[class_indices]
    mean_emb = np.mean(class_embs, axis=0)
    class_embeddings[class_name] = mean_emb

# Lưu embeddings trung bình
np.save('class_embeddings.npy', class_embeddings)
print("Đã lưu embeddings trung bình cho mỗi lớp.")
