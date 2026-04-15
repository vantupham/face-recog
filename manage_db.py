"""
manage_db.py - Quản lý cơ sở dữ liệu khuôn mặt
Cách dùng:
    python manage_db.py --list               # Liệt kê danh tính
    python manage_db.py --delete "Ten nguoi" # Xoá một danh tính
    python manage_db.py --clear              # Xoá toàn bộ CSDL
"""

import pickle
import os
import argparse
import numpy as np

EMBEDDINGS_FILE = "face_db.pkl"

def load_db():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("[INFO] Chưa có file CSDL.")
        return {}
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)

def save_db(db):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(db, f)

def list_identities():
    db = load_db()
    if not db:
        print("[INFO] CSDL trống.")
        return
    print(f"\n{'─'*40}")
    print(f"  CSDL khuôn mặt — {len(db)} danh tính")
    print(f"{'─'*40}")
    for i, (name, emb) in enumerate(db.items(), 1):
        print(f"  {i:2d}. {name:<30s}  emb shape: {emb.shape}")
    print(f"{'─'*40}\n")

def delete_identity(name):
    db = load_db()
    if name not in db:
        print(f"[WARN] Không tìm thấy '{name}' trong CSDL.")
        return
    del db[name]
    save_db(db)
    print(f"[OK] Đã xoá '{name}'. Còn lại {len(db)} danh tính.")

def clear_db():
    confirm = input("[!] Xác nhận xoá TOÀN BỘ CSDL? (yes/no): ").strip().lower()
    if confirm == "yes":
        save_db({})
        print("[OK] Đã xoá toàn bộ CSDL.")
    else:
        print("[INFO] Đã huỷ.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quản lý CSDL khuôn mặt")
    parser.add_argument("--list",   action="store_true",    help="Liệt kê danh tính")
    parser.add_argument("--delete", type=str, default=None, help="Xoá một danh tính")
    parser.add_argument("--clear",  action="store_true",    help="Xoá toàn bộ CSDL")
    args = parser.parse_args()

    if args.list:
        list_identities()
    elif args.delete:
        delete_identity(args.delete)
    elif args.clear:
        clear_db()
    else:
        parser.print_help()
