# predict_all.py (Versi Modifikasi & Revisi)
import cv2
import tensorflow as tf  # <-- REVISI 1: Typo 'tensortflow' diperbaiki
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from color_detector import get_dominant_color_isolated 
from image_enhancer import enhance_image_quality 

# --- 1. KONFIGURASI (Sama) ---
GENDER_INPUT = 'pria' 
# REVISI 2: Menggunakan forward slash (/) agar aman di semua OS
IMAGE_PATH = 'data/images/download (9).jpeg' 
STYLE_MODEL_PATH = 'model_style.h5'
MEN_MODEL_PATH = 'model_kategori_pria.h5'
WOMEN_MODEL_PATH = 'model_kategori_wanita.h5'

STYLE_CLASSES = ['Casual', 'Formal', 'Sport']
MEN_CLASSES = ['jacket', 'outer', 'pants', 'shirt', 'suit', 'tshirt']
WOMEN_CLASSES = ['jacket', 'blazer', 'blouse', 'dress', 'outer', 'pants', 'shirt', 'skirt', 'tshirt']

VALID_RULES = {
    'pria': {
        'Formal': ['suit', 'shirt', 'pants', 'tshirt'],
        'Casual': ['outer', 'pants', 'shirt', 'tshirt'],
        'Sport': ['jacket', 'pants', 'tshirt']
    },
    'wanita': {
        'Formal': ['blazer', 'blouse', 'dress', 'pants', 'shirt', 'skirt', 'tshirt', 'outer'],
        'Casual': ['blouse', 'outer', 'pants', 'shirt', 'skirt', 'tshirt'],
        'Sport': ['jacket', 'pants', 'tshirt', 'skirt']
    }
}

IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- 3. Fungsi Load & Prep (Sama) ---
def load_and_prep_image(image_array):
    img_display = image_array.copy()
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_processed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_processed, axis=0)
    return img_batch, img_display

# --- 4. Fungsi Prediksi ML (Sama) ---
def get_prediction(model, img_batch, class_names):
    predictions = model.predict(img_batch, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence

# --- 5. Load Model (Sama) ---
print("Memuat semua model...")
try:
    model_style = tf.keras.models.load_model(STYLE_MODEL_PATH)
    model_men = tf.keras.models.load_model(MEN_MODEL_PATH)
    model_women = tf.keras.models.load_model(WOMEN_MODEL_PATH)
    print("✅ Semua model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# --- 6. ALUR KERJA PREDIKSI (Sama) ---
print(f"\n🚀 Memproses gambar: {IMAGE_PATH}")
print(f"👤 Input Gender (dari login): {GENDER_INPUT}")

img_original = cv2.imread(IMAGE_PATH)
if img_original is None:
    print(f"Error: Gagal memuat gambar {IMAGE_PATH}")
    exit()

print("✨ Menerapkan Peningkatan Kualitas Gambar (CLAHE)...")
img_enhanced = enhance_image_quality(img_original)
image_batch, image_display = load_and_prep_image(img_enhanced)

print("🤖 Memprediksi Style...")
pred_style, conf_style = get_prediction(model_style, image_batch, STYLE_CLASSES)

print("🤖 Memprediksi Kategori...")
if GENDER_INPUT.lower() == 'pria':
    pred_kategori, conf_kategori = get_prediction(model_men, image_batch, MEN_CLASSES)
elif GENDER_INPUT.lower() == 'wanita':
    pred_kategori, conf_kategori = get_prediction(model_women, image_batch, WOMEN_CLASSES)
else:
    pred_kategori, conf_kategori = "Gender?", 0

print("🎨 Mendeteksi warna dominan (dengan isolasi objek)...")
pred_warna = get_dominant_color_isolated(img_original)

# --- 7. LOGIKA VALIDASI BARU (Sama) ---
print("\n🔍 Memvalidasi kombinasi...")
final_pred_style = pred_style
final_pred_kategori = pred_kategori

try:
    valid_categories_for_style = VALID_RULES[GENDER_INPUT.lower()][pred_style]
    
    if pred_kategori not in valid_categories_for_style:
        print(f"⚠ Peringatan: Konflik Logika!")
        print(f"   Model Style bilang: '{pred_style}'")
        print(f"   Model Kategori bilang: '{pred_kategori}'")
        print(f"   Aturan dataset: '{pred_style}' seharusnya hanya berisi {valid_categories_for_style}")
        
        if conf_style > conf_kategori:
            final_pred_kategori = f"<{pred_kategori}?>"
            print(f"   Keputusan: Percaya Style '{pred_style}'. Kategori mungkin tidak cocok.")
        else:
            final_pred_style = f"<{pred_style}?>"
            print(f"   Keputusan: Percaya Kategori '{pred_kategori}'. Style mungkin tidak cocok.")
            
    else:
        print("   ✅ Kombinasi Style dan Kategori valid.")

except KeyError:
    print(f"Error: Tidak ada aturan validasi untuk {GENDER_INPUT} atau {pred_style}")

# --- 8. Tampilkan Hasil (Sama) ---
hasil_akhir_full = f"{final_pred_style} {final_pred_kategori} {pred_warna}"
print("\n--- 🏁 HASIL AKHIR ---")
print(f"Style    : {final_pred_style} (Raw: {conf_style*100:.2f}%)")
print(f"Kategori : {final_pred_kategori} (Raw: {conf_kategori*100:.2f}%)")
print(f"Warna    : {pred_warna}")
print(f"Prediksi Final: {hasil_akhir_full}")

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image_display, hasil_akhir_full, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Hasil Prediksi (Enhanced Image)", image_display)
print("\nTekan tombol apapun di jendela gambar untuk keluar...")
cv2.waitKey(0)
cv2.destroyAllWindows()