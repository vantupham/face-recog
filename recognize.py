"""
recognize.py - Nhận diện khuôn mặt qua webcam sử dụng FaceNet (facenet-pytorch)
Chạy: python recognize.py
"""

import cv2
import torch
import numpy as np
import pickle
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# ── Cấu hình ──────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "face_db.pkl"   # File lưu embedding đã đăng ký
THRESHOLD       = 0.75            # Ngưỡng cosine similarity (0–1), tăng = nghiêm hơn
FRAME_SKIP      = 2               # Xử lý 1 frame mỗi N frame (tăng FPS)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Khởi tạo model ────────────────────────────────────────────────────────────
print(f"[INFO] Sử dụng thiết bị: {DEVICE.upper()}")

mtcnn   = MTCNN(keep_all=True, device=DEVICE, min_face_size=60,
                thresholds=[0.6, 0.7, 0.7])
resnet  = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ── Load cơ sở dữ liệu khuôn mặt ─────────────────────────────────────────────
def load_face_db():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("[WARN] Chưa có dữ liệu khuôn mặt. Hãy chạy register.py trước!")
        return {}, []
    with open(EMBEDDINGS_FILE, "rb") as f:
        db = pickle.load(f)
    names      = list(db.keys())
    embeddings = np.vstack([db[n] for n in names])  # (N, 512)
    print(f"[INFO] Đã load {len(names)} danh tính: {names}")
    return db, names, embeddings

db, names, known_embeddings = load_face_db() if os.path.exists(EMBEDDINGS_FILE) \
                               else ({}, [], np.empty((0, 512)))

# ── Nhận diện một khuôn mặt đã crop ──────────────────────────────────────────
def identify_face(face_tensor):
    """face_tensor: (1,3,160,160) float32 trên DEVICE"""
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy()          # (1,512)
    if known_embeddings.shape[0] == 0:
        return "Unknown", 0.0
    sims   = cosine_similarity(emb, known_embeddings)[0] # (N,)
    idx    = np.argmax(sims)
    score  = float(sims[idx])
    label  = names[idx] if score >= THRESHOLD else "Unknown"
    return label, score

# ── Vẽ bounding box ───────────────────────────────────────────────────────────
COLOR_KNOWN   = (0, 220, 80)
COLOR_UNKNOWN = (0, 80, 220)

def draw_box(frame, box, label, score):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = COLOR_KNOWN if label != "Unknown" else COLOR_UNKNOWN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text  = f"{label}  {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

# ── Vòng lặp chính ────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Đang chạy — nhấn Q để thoát, R để reload CSDL khuôn mặt")

    frame_count = 0
    last_results = []   # cache kết quả nhận diện frame trước

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        if frame_count % FRAME_SKIP == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            boxes, probs = mtcnn.detect(img)
            last_results = []

            if boxes is not None:
                faces = mtcnn.extract(img, boxes, save_path=None)  # (N,3,160,160)
                for i, (box, face_tensor) in enumerate(zip(boxes, faces)):
                    if face_tensor is None:
                        continue
                    face_input = face_tensor.unsqueeze(0).to(DEVICE)
                    label, score = identify_face(face_input)
                    last_results.append((box, label, score))

        # Vẽ kết quả cache lên frame hiển thị
        for box, label, score in last_results:
            draw_box(display, box, label, score)

        # HUD thông tin
        cv2.putText(display,
                    f"Faces: {len(last_results)}  |  Device: {DEVICE.upper()}  |  [Q] Quit  [R] Reload",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("FaceNet Recognition", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            global db, names, known_embeddings
            db, names, known_embeddings = load_face_db()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát.")

if __name__ == "__main__":
    main()
