# color_detector.py (PRO VERSION – RULE BASED FASHION COLOR)
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# =========================
# COLOR RANGE (HSV)
# =========================
COLOR_RANGES = {
    "Hitam":        ((0, 0, 0), (180, 255, 50)),
    "Putih":        ((0, 0, 200), (180, 40, 255)),
    "Abu-abu":      ((0, 0, 60), (180, 40, 200)),

    "Merah":        ((0, 50, 50), (10, 255, 255)),
    "Maroon":       ((0, 80, 40), (10, 255, 120)),
    "Burgundy":     ((160, 50, 50), (180, 255, 150)),

    "Oranye":       ((11, 80, 80), (20, 255, 255)),
    "Terracotta":  ((10, 60, 60), (18, 200, 200)),

    "Kuning":       ((21, 80, 80), (30, 255, 255)),
    "Mustard":      ((21, 60, 60), (30, 200, 200)),

    "Hijau":        ((36, 50, 50), (70, 255, 255)),
    "Olive":        ((36, 40, 40), (70, 150, 150)),
    "Sage":         ((36, 20, 80), (70, 80, 200)),
    "Mint":         ((36, 30, 150), (70, 120, 255)),

    "Biru":         ((90, 50, 50), (130, 255, 255)),
    "Navy":         ((100, 80, 40), (130, 255, 120)),
    "Denim":        ((90, 50, 80), (120, 200, 200)),

    "Ungu":         ((131, 50, 50), (160, 255, 255)),
    "Lavender":     ((131, 30, 120), (160, 120, 255)),

    "Pink":         ((160, 50, 80), (180, 255, 255)),
    "Dusty Pink":   ((160, 20, 120), (180, 120, 200)),

    "Coklat":       ((10, 50, 20), (20, 255, 150)),
    "Tan":          ((10, 30, 120), (20, 150, 200)),
    "Beige":        ((15, 20, 180), (30, 80, 255)),
    "Cream":        ((15, 10, 200), (30, 60, 255)),
    "Ivory":        ((0, 0, 220), (30, 40, 255)),
}

# =========================
# MASK BACKGROUND PUTIH
# =========================
def isolate_clothing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return mask

# =========================
# DOMINANT COLOR
# =========================
def get_dominant_color_isolated(image, k=4):
    try:
        # [OPTIMASI 1] Resize gambar jadi kecil banget sebelum diproses
        # K-Means pada gambar 100px sama akuratnya dengan 4000px, tapi jauh lebih cepat
        h, w = image.shape[:2]
        if h > 200 or w > 200: 
            # Resize proporsional ke max 150px
            scale = 150 / max(h, w)
            image = cv2.resize(image, (0,0), fx=scale, fy=scale)

        mask = isolate_clothing(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        pixels = hsv.reshape((-1, 3))
        mask_flat = mask.reshape((-1,))

        clothing_pixels = pixels[mask_flat != 0]
        if len(clothing_pixels) == 0:
            return "Tidak terdeteksi"

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3) # n_init turunkan ke 3 biar cepat
        kmeans.fit(clothing_pixels)

        counts = Counter(kmeans.labels_)
        dominant_cluster = counts.most_common(1)[0][0]
        dominant_hsv = kmeans.cluster_centers_[dominant_cluster]

        return classify_color_hsv(dominant_hsv)

    except Exception as e:
        print(f"[ColorDetector ERROR] {e}")
        return "Error"
    
# =========================
# HSV → FASHION COLOR
# =========================
def classify_color_hsv(hsv_pixel):
    h, s, v = hsv_pixel

    for color_name, (lower, upper) in COLOR_RANGES.items():
        if (
            lower[0] <= h <= upper[0] and
            lower[1] <= s <= upper[1] and
            lower[2] <= v <= upper[2]
        ):
            return color_name

    return "Warna Lain"

def get_dominant_color_global(image, k=3):
    # [OPTIMASI 2] Resize global juga
    h, w = image.shape[:2]
    if h > 200 or w > 200: 
        scale = 150 / max(h, w)
        image = cv2.resize(image, (0,0), fx=scale, fy=scale)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(pixels)

    dominant = np.argmax(np.bincount(kmeans.labels_))
    dominant_hsv = kmeans.cluster_centers_[dominant]

    return classify_color_hsv(dominant_hsv)