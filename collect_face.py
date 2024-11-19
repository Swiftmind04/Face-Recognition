
import numpy as np
import cv2
import os
import torch
from facenet_pytorch import MTCNN

def collect_faces(name):
    # Khởi tạo thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Khởi tạo MTCNN
    mtcnn = MTCNN(keep_all=False, image_size=160, margin=0, device=device)

    cap = cv2.VideoCapture(0)
    count = 0
    # Lưu ảnh đã căn chỉnh trực tiếp vào aligned_dataset
    os.makedirs(f'aligned_dataset/{name}', exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện và căn chỉnh khuôn mặt bằng MTCNN
        img_cropped = mtcnn(frame)
        if img_cropped is not None:
            # Chuyển đổi tensor sang NumPy array
            img_cropped = img_cropped.permute(1, 2, 0).numpy()
            # Nhân với 255 và chuyển đổi sang uint8
            img_cropped = (img_cropped * 255).astype(np.uint8)
            # Chuyển từ RGB sang BGR để OpenCV xử lý đúng màu
            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)
            # Lưu ảnh đã căn chỉnh vào aligned_dataset
            img_path = f'aligned_dataset/{name}/{count}.jpg'
            cv2.imwrite(img_path, img_cropped)
            count += 1

            print(f"Đã lưu ảnh: {img_path}")

            # Vẽ khung và hiển thị số lượng ảnh đã thu thập
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Images Captured: {count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if count >= 300:
                break

        cv2.imshow('Collecting Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập {count} ảnh cho người dùng '{name}'.")

if __name__ == "__main__":
    name = input("Nhập tên của bạn: ")
    collect_faces(name)
