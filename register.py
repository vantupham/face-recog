"""
register.py - Đăng ký khuôn mặt mới vào cơ sở dữ liệu
Cách dùng:
    python register.py --name "Nguyen Van A" --samples 30
    python register.py --name "Tran Thi B" --samples 30
"""

import cv2
import torch
import numpy as np
import pickle
import os
import argparse
import time
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ── Cấu hình ──────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "face_db.pkl"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Khởi tạo model ────────────────────────────────────────────────────────────
print(f"[INFO] Thiết bị: {DEVICE.upper()}")
mtcnn  = MTCNN(keep_all=False, device=DEVICE, min_face_size=80,
               thresholds=[0.6, 0.7, 0.7], select_largest=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ── Load / Khởi tạo CSDL ──────────────────────────────────────────────────────
def load_db():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(db, f)
    print(f"[INFO] Đã lưu CSDL: {EMBEDDINGS_FILE}")

# ── Tính embedding ────────────────────────────────────────────────────────────
def get_embedding(face_tensor):
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(DEVICE))
    return emb.cpu().numpy()   # (1, 512)

# ── Chương trình đăng ký ──────────────────────────────────────────────────────
def register(name: str, num_samples: int):
    db = load_db()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\n[INFO] Đang đăng ký: '{name}'")
    print(f"[INFO] Sẽ thu {num_samples} mẫu.")
    print("[INFO] Nhìn thẳng vào camera. Nhấn SPACE để bắt đầu, Q để huỷ.\n")

    embeddings_collected = []
    capturing   = False
    countdown   = 3
    last_tick   = time.time()
    sample_interval = 0.15   # giây giữa 2 lần chụp

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Phát hiện khuôn mặt
        box, prob = mtcnn.detect(img)
        face_found = (box is not None and prob[0] is not None and prob[0] > 0.90)

        if face_found:
            x1, y1, x2, y2 = [int(v) for v in box[0]]
            color = (0, 220, 80) if capturing else (255, 200, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # ── Countdown trước khi chụp ──────────────────────────────────────────
        if capturing and time.time() - last_tick >= 1.0 and countdown > 0:
            countdown  -= 1
            last_tick   = time.time()

        # ── Thu mẫu ───────────────────────────────────────────────────────────
        if capturing and countdown == 0 and face_found:
            if time.time() - last_tick >= sample_interval:
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    emb = get_embedding(face_tensor)
                    embeddings_collected.append(emb)
                    last_tick = time.time()
                    print(f"  [{len(embeddings_collected)}/{num_samples}] Đã thu mẫu", end="\r")

        # ── Hoàn thành ────────────────────────────────────────────────────────
        if capturing and len(embeddings_collected) >= num_samples:
            break

        # ── HUD ───────────────────────────────────────────────────────────────
        if not capturing:
            status = f"[SPACE] Bat dau  |  '{name}'  |  Face: {'OK' if face_found else 'Khong tim thay'}"
        elif countdown > 0:
            status = f"Chuan bi... {countdown}"
        else:
            status = f"Dang thu: {len(embeddings_collected)}/{num_samples}"

        cv2.putText(display, status, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        # Thanh tiến độ
        if capturing and countdown == 0:
            pct = len(embeddings_collected) / num_samples
            bar_w = int(frame.shape[1] * pct)
            cv2.rectangle(display, (0, frame.shape[0]-8), (bar_w, frame.shape[0]),
                          (0, 220, 80), -1)

        cv2.imshow("Dang ky khuon mat", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not capturing:
            capturing = True
            last_tick = time.time()
        elif key == ord("q"):
            print("\n[INFO] Đã huỷ.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings_collected) == 0:
        print("\n[WARN] Không thu được mẫu nào.")
        return

    # Lưu embedding trung bình của tất cả mẫu
    avg_emb = np.mean(np.vstack(embeddings_collected), axis=0, keepdims=True)
    db[name] = avg_emb
    save_db(db)

    print(f"\n[OK] Đã đăng ký thành công '{name}' với {len(embeddings_collected)} mẫu.")
    print(f"[INFO] CSDL hiện có {len(db)} danh tính: {list(db.keys())}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đăng ký khuôn mặt vào CSDL FaceNet")
    parser.add_argument("--name",    type=str, required=True,  help="Tên người đăng ký")
    parser.add_argument("--samples", type=int, default=30,     help="Số mẫu cần thu (mặc định 30)")
    args = parser.parse_args()

    register(args.name.strip(), args.samples)
