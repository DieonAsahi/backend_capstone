# predict_skin.py
import cv2
import numpy as np
import pickle
import os

# ==============================
# LOAD MODEL (GLOBAL)
# ==============================
SKIN_MODEL_PATH = 'skin_tone_model.pkl'
skin_model = None

if os.path.exists(SKIN_MODEL_PATH):
    with open(SKIN_MODEL_PATH, 'rb') as f:
        skin_model = pickle.load(f)
    print("✅ Model Warna Kulit berhasil dimuat")
else:
    print("❌ Model skin_tone_model.pkl tidak ditemukan")

# Load Haar Cascade (sekali saja)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

SCALER_PATH = 'skin_scaler.pkl'
skin_scaler = None

if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        skin_scaler = pickle.load(f)
    print("✅ Scaler berhasil dimuat")
else:
    print("❌ skin_scaler.pkl tidak ditemukan")

# ==============================
# FUNGSI DETEKSI WARNA KULIT
# ==============================
def detect_skin_tone_ai(image_path):
    if skin_model is None:
        return "Model Tidak Tersedia"

    img = cv2.imread(image_path)
    if img is None:
        return "Gambar Tidak Valid"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    if len(faces) == 0:
        return "Wajah Tidak Terdeteksi"

    # Ambil wajah TERBESAR
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # ==============================
    # AMBIL AREA TENGAH WAJAH (ANTI RAMBUT & BACKGROUND)
    # ==============================
    crop_x = x + int(w * 0.3)
    crop_y = y + int(h * 0.3)
    crop_w = int(w * 0.4)
    crop_h = int(h * 0.4)

    roi_skin = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    if roi_skin.size == 0:
        return "ROI Error"

    # ==============================
    # HITUNG WARNA RATA-RATA (RGB)
    # ==============================
    avg_color = np.mean(roi_skin.reshape(-1, 3), axis=0)
    b, g, r = avg_color

    rgb_input = [[r, g, b]]

    # ==============================
    # DEBUG RGB
    # ==============================
    print("RGB RAW:", r, g, b)

    # NORMALISASI (WAJIB)
    rgb_scaled = skin_scaler.transform(rgb_input)

    print("RGB SCALED:", rgb_scaled)

    # ==============================
    # PREDIKSI KNN
    # ==============================
    prediction = skin_model.predict(rgb_scaled)

    return prediction[0]
