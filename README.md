# 🎯 Nhận diện khuôn mặt bằng FaceNet — Hướng dẫn cài đặt & sử dụng

## Cấu trúc thư mục

```
facenet_recognition/
├── recognize.py      # Nhận diện qua webcam (chương trình chính)
├── register.py       # Đăng ký khuôn mặt mới
├── manage_db.py      # Quản lý cơ sở dữ liệu
├── requirements.txt  # Thư viện cần cài
└── face_db.pkl       # CSDL khuôn mặt (tự tạo sau khi đăng ký)
```

---

## Bước 1 — Cài đặt môi trường Python

Mở **Terminal** (hoặc PowerShell) trong VS Code (`Ctrl + \``):

```bash
# Nếu chạy lệnh để cấp quyền cho user
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

> **Lưu ý GPU (tuỳ chọn — tăng tốc 3–5x):**  
> Nếu máy có NVIDIA GPU, cài PyTorch với CUDA trước:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Sau đó mới chạy `pip install -r requirements.txt`

---

## Bước 2 — Đăng ký khuôn mặt

```bash
python register.py --name "Nguyen Van A" --samples 30
```

- Cửa sổ webcam sẽ mở ra
- Nhìn thẳng vào camera, nhấn **SPACE** để bắt đầu
- Chương trình tự động thu 30 mẫu khuôn mặt
- Lặp lại cho mỗi người cần nhận diện

**Gợi ý:** Đăng ký khuôn mặt ở các góc độ nhẹ khác nhau (trái, phải, ngước, cúi) để tăng độ chính xác. Chạy lệnh nhiều lần với cùng tên sẽ **cộng dồn** mẫu.

---

## Bước 3 — Chạy nhận diện

```bash
python recognize.py
```

| Phím | Chức năng |
|------|-----------|
| **Q** | Thoát chương trình |
| **R** | Tải lại CSDL khuôn mặt (khi đã đăng ký thêm người) |

---

## Quản lý cơ sở dữ liệu

```bash
# Liệt kê tất cả danh tính đã đăng ký
python manage_db.py --list

# Xoá một người
python manage_db.py --delete "Nguyen Van A"

# Xoá toàn bộ
python manage_db.py --clear
```

---

## Điều chỉnh độ nhạy

Mở `recognize.py`, tìm dòng:

```python
THRESHOLD = 0.75
```

| Giá trị | Ý nghĩa |
|---------|---------|
| `0.60` | Dễ nhận diện hơn, nhưng có thể nhầm lẫn |
| `0.75` | Cân bằng (mặc định) |
| `0.85` | Nghiêm ngặt hơn, ít nhầm nhưng dễ bị Unknown |

---

## Yêu cầu hệ thống

- Python **3.8 – 3.11**
- Windows 10/11
- Webcam (tích hợp hoặc USB)
- RAM ≥ 4GB (8GB+ khuyến nghị)
- GPU NVIDIA (tuỳ chọn, tăng tốc đáng kể)

---

## Cách hoạt động

```
Webcam frame
    │
    ▼
MTCNN (phát hiện & căn chỉnh khuôn mặt)
    │  → crop 160×160 px
    ▼
FaceNet / InceptionResnetV1 (trích xuất đặc trưng)
    │  → vector 512 chiều
    ▼
Cosine Similarity với CSDL
    │  → score 0–1
    ▼
Nếu score ≥ THRESHOLD → hiển thị tên
Nếu score < THRESHOLD → "Unknown"
```

---

## Xử lý lỗi thường gặp

| Lỗi | Giải pháp |
|-----|-----------|
| `No module named 'facenet_pytorch'` | Chạy lại `pip install -r requirements.txt` |
| Webcam không mở được | Kiểm tra camera trong Device Manager, thử `cv2.VideoCapture(1)` |
| FPS thấp | Tăng `FRAME_SKIP = 3` trong `recognize.py` |
| Nhận diện kém | Đăng ký thêm mẫu ở nhiều góc độ, điều chỉnh ánh sáng |
