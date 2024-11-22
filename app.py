from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_file
import cv2
import torch
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
import os
from PIL import Image
from scipy.spatial.distance import cosine
import threading
import time
import base64
from threading import Lock
from datetime import datetime
import csv
import io

app = Flask(__name__)

# Khởi tạo thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Khởi tạo mô hình FaceNet và MTCNN
model = InceptionResnetV1(pretrained='vggface2').to(device)
model.eval()
mtcnn = MTCNN(keep_all=False, image_size=160, margin=0, device=device)

# Biến toàn cục để lưu trạng thái
is_collecting = False
collected_images = []
collect_name = ""
is_training = False

# Chuẩn bị phép biến đổi
transformation = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Biến toàn cục để lưu thông tin nhận diện
recognized_faces = {}
recognition_lock = Lock()

# Hàm để tải embeddings và class_names
def load_embeddings():
    if os.path.exists('class_embeddings.npy') and os.path.exists('class_names.npy'):
        class_embeddings = np.load('class_embeddings.npy', allow_pickle=True).item()
        class_names = np.load('class_names.npy')
        return class_embeddings, class_names
    else:
        return {}, []

class_embeddings, class_names = load_embeddings()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect', methods=['GET', 'POST'])
def collect():
    global is_collecting, collected_images, collect_name
    if request.method == 'POST':
        collect_name = request.form.get('name')
        if not collect_name:
            return "Vui lòng nhập tên."
        os.makedirs(f'dataset/{collect_name}', exist_ok=True)
        is_collecting = True
        collected_images = []
        return render_template('collect.html', name=collect_name)
    else:
        return render_template('collect_name.html')

@app.route('/collect_frames', methods=['POST'])
def collect_frames():
    global is_collecting, collected_images, collect_name
    if not is_collecting:
        return jsonify({'status': 'not_collecting'})
    try:
        data = request.json
        img_data = data['image']
        img_str = base64.b64decode(img_data.split(',')[1])
        img_np = np.frombuffer(img_str, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # Phát hiện và căn chỉnh khuôn mặt bằng MTCNN
        img_cropped = mtcnn(frame)
        if img_cropped is not None:
            if isinstance(img_cropped, list) or img_cropped.ndim == 4:
                img_cropped = img_cropped[0]  # Lấy khuôn mặt đầu tiên
            # Chuyển đổi tensor sang NumPy array
            img_cropped = img_cropped.permute(1, 2, 0).numpy()
            # Nhân với 255 và chuyển đổi sang uint8
            img_cropped = (img_cropped * 255).astype(np.uint8)
            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)
            # Lưu ảnh đã căn chỉnh vào aligned_dataset
            os.makedirs(f'aligned_dataset/{collect_name}', exist_ok=True)
            img_path = f'aligned_dataset/{collect_name}/{len(collected_images)}.jpg'
            cv2.imwrite(img_path, img_cropped)
            collected_images.append(img_cropped)
            print(f"Đã lưu ảnh: {img_path}")
        if len(collected_images) >= 100:  # Thu thập 100 ảnh
            is_collecting = False
            return jsonify({'status': 'completed'})
        return jsonify({'status': 'collecting', 'count': len(collected_images)})
    except Exception as e:
        print(f"Lỗi trong collect_frames: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/train', methods=['GET'])
def train():
    global is_training, class_embeddings, class_names
    if is_training:
        return render_template('train.html', message="Đang huấn luyện, vui lòng chờ...")
    else:
        is_training = True
        threading.Thread(target=train_model).start()
        return render_template('train.html', message="Bắt đầu huấn luyện, vui lòng đợi...")

def train_model():
    global is_training, class_embeddings, class_names
    time.sleep(1)  # Đợi một chút để trả về phản hồi cho người dùng
    try:
        # Thực hiện quá trình huấn luyện
        data_dir = 'aligned_dataset'
        batch_size = 32
        transformation = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.ImageFolder(data_dir, transform=transformation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
    except Exception as e:
        print(f"Lỗi trong train_model: {e}")
    finally:
        is_training = False

@app.route('/train_status')
def train_status():
    global is_training
    if is_training:
        return jsonify({'status': 'training'})
    else:
        return jsonify({'status': 'completed'})

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

def gen_frames():
    global class_embeddings, class_names, recognized_faces
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    # Biến để tính FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Tính FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            fps = int(fps)

            # Hiển thị FPS
            cv2.putText(frame, f'FPS: {fps}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            img = frame.copy()
            # Phát hiện khuôn mặt bằng MTCNN
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    face_img = img[y1:y2, x1:x2]
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_img_pil = Image.fromarray(face_img_rgb)
                    face_tensor = transformation(face_img_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = model(face_tensor).cpu().numpy()[0]

                    # Tính khoảng cách đến embeddings trung bình của mỗi lớp
                    distances = []
                    for class_name, class_emb in class_embeddings.items():
                        distance = cosine(embedding, class_emb)
                        distances.append((distance, class_name))

                    # Sắp xếp theo khoảng cách tăng dần
                    distances.sort()
                    min_distance, label = distances[0]

                    # Ngưỡng để quyết định Unknown
                    threshold = 0.6  # Có thể cần điều chỉnh

                    if min_distance < threshold:
                        label = label
                        # Lưu thông tin nhận diện
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with recognition_lock:
                            recognized_faces[label] = current_time
                            print(f"Đã nhận diện: {label} vào lúc {current_time}")
                    else:
                        label = 'Unknown'

                    # Vẽ khung và hiển thị tên
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} ({min_distance:.2f})', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/collect_video_feed')
def collect_video_feed():
    return Response(gen_collect_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_collect_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    while is_collecting:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/recognition_info')
def recognition_info():
    with recognition_lock:
        # Chuyển dictionary thành list các dict
        recognized_list = [{'name': name, 'time': time} for name, time in recognized_faces.items()]
        return jsonify(recognized_list)

@app.route('/clear_recognition', methods=['POST'])
def clear_recognition():
    global recognized_faces
    try:
        with recognition_lock:
            recognized_faces.clear()
        return jsonify({'status': 'cleared'})
    except Exception as e:
        print(f"Lỗi trong clear_recognition: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/export_data', methods=['GET'])
def export_data():
    try:
        with recognition_lock:
            recognized_list = [{'name': name, 'time': time} for name, time in recognized_faces.items()]
        
        # Tạo một đối tượng StringIO để ghi dữ liệu CSV vào bộ nhớ
        si = io.StringIO()
        writer = csv.DictWriter(si, fieldnames=['Name', 'Time'])
        writer.writeheader()
        for entry in recognized_list:
            writer.writerow({'Name': entry['name'], 'Time': entry['time']})
        
        # Chuyển đổi StringIO thành BytesIO để có thể gửi dưới dạng file tải xuống
        mem = io.BytesIO()
        mem.write(si.getvalue().encode('utf-8'))
        mem.seek(0)
        
        # Sửa từ 'attachment_filename' sang 'download_name'
        return send_file(mem,
                        mimetype='text/csv',
                        download_name='data.csv',
                        as_attachment=True)
    except Exception as e:
        print(f"Lỗi trong export_data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
