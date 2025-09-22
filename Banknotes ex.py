import os, sys, json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
from tensorflow import keras
import joblib

# --------------------------
# Paths & feature config
# --------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_ann.h5")
LABELS_PATH = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_labels.json")
SCALER_PATH = os.path.join(SCRIPT_DIR, "vn_banknotes_color_pixel_scaler.pkl")

PIXEL_SIZE  = (16, 16)
RESIZE_BIG  = (64, 64)

# --------------------------
# Load assets
# --------------------------
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    if isinstance(data, list):
        return data
    raise ValueError("Labels JSON không hợp lệ")

def robust_imread_bgr(path: str):
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def extract_feature_from_path(path):
    img = robust_imread_bgr(path)
    if img is None:
        return None
    big = cv2.resize(img, RESIZE_BIG).astype("float32") / 255.0
    mean_col = big.mean(axis=(0, 1))  # B,G,R
    std_col  = big.std(axis=(0, 1))
    small = cv2.resize(img, PIXEL_SIZE).astype("float32") / 255.0
    flat_pixels = small.flatten()
    feat = np.concatenate([mean_col, std_col, flat_pixels]).astype("float32")
    return feat.reshape(1, -1)

# Load
try:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)   # 🔑 dùng lại scaler khi train
    class_names = load_labels(LABELS_PATH)
except Exception as e:
    messagebox.showerror("Lỗi", f"Không thể load tài nguyên:\n{e}")
    sys.exit(1)

# --------------------------
# GUI
# --------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("💵 Vietnamese Banknote Detector")
        self.root.geometry("980x640")
        self.root.configure(bg="#eaf2ff")

        header = tk.Label(root, text="💵 Banknote Recognition",
                          font=("Segoe UI", 22, "bold"), fg="#0b63c4", bg="#eaf2ff")
        header.pack(pady=12)

        body = tk.Frame(root, bg="#eaf2ff")
        body.pack(fill="both", expand=True, padx=16, pady=6)

        self.preview_card = tk.Frame(body, bg="white", highlightthickness=1, highlightbackground="#d5e3ff")
        self.preview_card.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=6)
        tk.Label(self.preview_card, text="Ảnh xem trước", font=("Segoe UI", 12, "bold"),
                 bg="white", fg="#124a8a").pack(anchor="w", padx=16, pady=(12, 6))

        self.preview = tk.Label(self.preview_card, bg="#f6f7fb", bd=1, relief="solid")
        self.preview.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.preview.configure(text="Chưa có ảnh.\nNhấn '📂 Chọn ảnh' để bắt đầu.",
                               font=("Segoe UI", 12), fg="#5a5a5a")

        ctrl = tk.Frame(body, width=340, bg="white", highlightthickness=1, highlightbackground="#d5e3ff")
        ctrl.pack(side="right", fill="y", padx=(8, 0), pady=6)
        tk.Label(ctrl, text="Điều khiển", font=("Segoe UI", 12, "bold"),
                 bg="white", fg="#124a8a").pack(anchor="w", padx=16, pady=(12, 6))

        btns = tk.Frame(ctrl, bg="white"); btns.pack(fill="x", padx=16, pady=(0,10))
        tk.Button(btns, text="📂 Chọn ảnh", command=self.on_select,
                  font=("Segoe UI", 11, "bold"), fg="white", bg="#0b63c4",
                  activebackground="#0a54a6", relief="flat", height=2).pack(fill="x", pady=(0,10))
        tk.Button(btns, text="🔍 Detect", command=self.on_detect,
                  font=("Segoe UI", 11, "bold"), fg="white", bg="#1db954",
                  activebackground="#169e47", relief="flat", height=2).pack(fill="x", pady=(0,10))
        tk.Button(btns, text="🗑 Xóa", command=self.on_clear,
                  font=("Segoe UI", 11, "bold"), fg="white", bg="#e53935",
                  activebackground="#c62828", relief="flat", height=2).pack(fill="x")

        res = tk.Frame(ctrl, bg="#f9fbff"); res.pack(fill="x", padx=16, pady=(10,16))
        tk.Label(res, text="Kết quả", font=("Segoe UI", 12, "bold"),
                 bg="#f9fbff", fg="#124a8a").pack(anchor="w")
        self.var_result = tk.StringVar(value="Chưa có kết quả.")
        tk.Label(res, textvariable=self.var_result, font=("Segoe UI", 12),
                 bg="#f9fbff", fg="#1a1a1a", justify="left").pack(anchor="w", pady=(6,0))

        self.status = tk.StringVar(value="Sẵn sàng.")
        tk.Label(root, textvariable=self.status, anchor="w",
                 font=("Segoe UI", 10), bg="#dfe9ff", fg="#334e73").pack(fill="x", side="bottom")

        self.image_path = None
        self._tkimg = None

    def on_select(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh tiền",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png")]
        )
        if not path: return
        self.image_path = path
        img = Image.open(path).convert("RGB")
        w = self.preview.winfo_width() or 640
        h = self.preview.winfo_height() or 360
        img = ImageOps.contain(img, (w-20, h-20))
        self._tkimg = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._tkimg, text="")
        self.var_result.set("Chưa có kết quả.")
        self.status.set(f"Đã chọn: {os.path.basename(path)}")

    def on_detect(self):
        if not self.image_path:
            messagebox.showwarning("Thiếu ảnh", "Bạn chưa chọn ảnh!")
            return
        feat = extract_feature_from_path(self.image_path)
        if feat is None:
            messagebox.showerror("Lỗi detect", "Không đọc được ảnh (đường dẫn/định dạng).")
            return
        # 🔑 Áp dụng scaler giống khi train
        feat_scaled = scaler.transform(feat)
        preds = model.predict(feat_scaled, verbose=0)
        idx = int(np.argmax(preds))
        prob = float(np.max(preds)) * 100.0
        cls = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        self.var_result.set(f"💰 Dự đoán: {cls} VND\n📊 Độ tin cậy: {prob:.2f}%")
        self.status.set("Detect xong.")

    def on_clear(self):
        self.image_path = None
        self.preview.configure(image="", text="Chưa có ảnh.\nNhấn '📂 Chọn ảnh' để bắt đầu.",
                               font=("Segoe UI", 12), fg="#5a5a5a", bg="#f6f7fb")
        self._tkimg = None
        self.var_result.set("Chưa có kết quả.")
        self.status.set("Đã xóa.")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
