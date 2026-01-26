import cv2
import numpy as np

def calculate_brightness(image):
    """
    Menghitung rata-rata kecerahan gambar.
    Menggunakan konversi ke HSV dan mengambil rata-rata channel V (Value).
    Range: 0 (Gelap Gulita) - 255 (Terang Benderang).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return np.mean(v)

def apply_gamma_correction(image, gamma=1.0):
    """
    Metode 1: Gamma Correction.
    Sangat bagus untuk mencerahkan gambar gelap tanpa membuat warna jadi pucat.
    gamma < 1.0 = Mencerahkan (Cocok untuk kasus baju hijau yg jadi hitam)
    gamma > 1.0 = Menggelapkan
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_log_transform(image):
    """
    Metode 2: Log Transform.
    Memperluas nilai gelap menjadi lebih terang.
    """
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    return np.array(log_image, dtype=np.uint8)

def apply_clahe(image):
    """
    Metode 3: CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Bagus untuk detail tekstur, tapi hati-hati di warna.
    Kita terapkan hanya pada channel Luminance (L) di LAB color space agar warna aman.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def smart_enhance(image):
    """
    Fungsi Otomatis:
    - Jika Gelap -> Pakai Gamma Correction (Biar warna keluar)
    - Jika Normal -> Pakai CLAHE (Biar tekstur tajam)
    """
    brightness = calculate_brightness(image)
    print(f"💡 Tingkat Kecerahan Original: {brightness:.2f}/255")

    # THRESHOLD: Jika kecerahan di bawah 80, dianggap GELAP.
    if brightness < 80:
        print("   👉 Gambar Gelap terdeteksi! Menerapkan Gamma Correction (Mencerahkan)...")
        # Gamma 0.5 artinya mencerahkan secara signifikan
        enhanced = apply_gamma_correction(image, gamma=0.5) 
        
        # Cek lagi setelah dicerahkan
        new_b = calculate_brightness(enhanced)
        print(f"   💡 Kecerahan Baru: {new_b:.2f}/255")
        return enhanced
        
    else:
        print("   👉 Pencahayaan Cukup. Menerapkan CLAHE (Pertajam Tekstur)...")
        return apply_clahe(image)