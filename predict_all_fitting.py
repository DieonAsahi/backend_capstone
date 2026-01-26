import cv2
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from color_detector import get_dominant_color_isolated, get_dominant_color_global
# [PERBAIKAN] Gunakan smart_enhance, bukan enhance_image_quality
from image_enhancer import smart_enhance 

# --- 1. KONFIGURASI INPUT ---
IMAGE_PATH = 'data/images/2.jpg' 
GENDER_INPUT = 'pria'  # Ganti 'wanita' jika ingin tes baju cewek

# --- 2. KONFIGURASI MODEL UTAMA ---
STYLE_MODEL_PATH = 'model_style.h5'
MEN_MODEL_PATH = 'model_kategori_pria.h5'
WOMEN_MODEL_PATH = 'model_kategori_wanita.h5'

STYLE_CLASSES = ['Casual', 'Formal', 'Sport']
MEN_CLASSES = ['jacket', 'outer', 'pants', 'shirt', 'suit', 'tshirt']
WOMEN_CLASSES = ['jacket', 'blazer', 'blouse', 'dress', 'outer', 'pants', 'shirt', 'skirt', 'tshirt']

# --- 3. KONFIGURASI MODEL FITTING ---
FIT_CONFIGS = {
    'fit_pria_baju':   ('model_fit_pria_baju.h5',   ['Fit', 'Loose']),
    'fit_pria_celana': ('model_fit_pria_celana.h5', ['Fit', 'Loose']),
    'fit_wanita_baju': ('model_fit_wanita_baju.h5', ['Fitted', 'Flare', 'Loose', 'Shoulder']),
    'fit_wanita_celana':('model_fit_wanita_celana.h5',['Fitted', 'Flare', 'Straight', 'Wide']),
    'fit_wanita_rok':  ('model_fit_wanita_rok.h5',  ['Flare', 'Mermaid', 'Straight'])
}

IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- 4. FUNGSI HELPER ---
def load_and_prep_image(image_array):
    img_display = image_array.copy()
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_processed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_processed, axis=0)
    return img_batch, img_display

def get_prediction(model, img_batch, class_names):
    if model is None: return "Model Error", 0.0
    predictions = model.predict(img_batch, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence

# --- 5. MEMUAT SEMUA MODEL ---
print("⏳ Sedang memuat model AI...")
try:
    model_style = tf.keras.models.load_model(STYLE_MODEL_PATH)
    model_men = tf.keras.models.load_model(MEN_MODEL_PATH)
    model_women = tf.keras.models.load_model(WOMEN_MODEL_PATH)
    
    loaded_fit_models = {}
    for key, (path, labels) in FIT_CONFIGS.items():
        if os.path.exists(path):
            loaded_fit_models[key] = tf.keras.models.load_model(path)
            print(f"   ✅ Loaded: {key}")
        else:
            print(f"   ❌ Gagal load {key}")
            loaded_fit_models[key] = None
    print("✅ Load selesai.\n")
except Exception as e:
    print(f"FATAL ERROR saat load model: {e}")
    exit()

# --- 6. EKSEKUSI PREDIKSI ---
print(f"🚀 Memproses gambar: {IMAGE_PATH}")
img_original = cv2.imread(IMAGE_PATH)
if img_original is None:
    print("Error: Gambar tidak ditemukan!")
    exit()

# [PERBAIKAN] Panggil Smart Enhance
print("💡 Menerapkan Smart Enhancement (Gamma/CLAHE)...")
img_enhanced = smart_enhance(img_original)

image_batch, image_display = load_and_prep_image(img_enhanced)

# A. Prediksi Style
print("1️⃣  Mendeteksi Style...")
pred_style, conf_style = get_prediction(model_style, image_batch, STYLE_CLASSES)

# B. Prediksi Kategori
print("2️⃣  Mendeteksi Kategori...")
if GENDER_INPUT.lower() == 'pria':
    pred_kategori, conf_kategori = get_prediction(model_men, image_batch, MEN_CLASSES)
else:
    pred_kategori, conf_kategori = get_prediction(model_women, image_batch, WOMEN_CLASSES)

# C. Prediksi FITTING (LOGIKA TERBARU)
print("3️⃣  Mendeteksi Fitting...")
pred_fitting = "-"
fit_model_key = None
cat_lower = pred_kategori.lower()

# List Sesuai Request Anda
list_baju_wanita = ['blouse', 'outer', 'shirt', 'tshirt', 'blazer', 'dress', 'jacket']
list_baju_pria = ['shirt', 'tshirt', 'outer', 'jacket', 'suit'] 

if GENDER_INPUT.lower() == 'pria':
    if cat_lower in list_baju_pria:
        fit_model_key = 'fit_pria_baju'
    elif cat_lower in ['pants']:
        fit_model_key = 'fit_pria_celana'
        
elif GENDER_INPUT.lower() == 'wanita':
    if cat_lower in list_baju_wanita:
        fit_model_key = 'fit_wanita_baju'
    elif cat_lower in ['pants']:
        fit_model_key = 'fit_wanita_celana'
    elif cat_lower in ['skirt']:
        fit_model_key = 'fit_wanita_rok'

if fit_model_key and loaded_fit_models.get(fit_model_key):
    fit_model = loaded_fit_models[fit_model_key]
    fit_labels = FIT_CONFIGS[fit_model_key][1]
    pred_fitting, conf_fit = get_prediction(fit_model, image_batch, fit_labels)
    print(f"   🎯 Fitting Model: {fit_model_key}")
else:
    pred_fitting = "-"
    conf_fit = 0.0

# D. Prediksi Warna (Pakai Enhanced Image!)
print("4️⃣  Mendeteksi Warna...")
pred_warna = get_dominant_color_isolated(img_enhanced)
if not pred_warna or pred_warna in ['Error', 'Tidak terdeteksi']:
    pred_warna = get_dominant_color_global(img_enhanced)

# --- 7. TAMPILKAN HASIL ---
print("\n" + "="*40)
print(f"      HASIL ANALISA AI ({GENDER_INPUT.upper()})")
print("="*40)
print(f"🧥 Style    : {pred_style} ({conf_style*100:.1f}%)")
print(f"👕 Kategori : {pred_kategori} ({conf_kategori*100:.1f}%)")
print(f"✂️  Fitting  : {pred_fitting} ({conf_fit*100:.1f}%)")
print(f"🎨 Warna    : {pred_warna}")
print("="*40)

text1 = f"{pred_style} - {pred_kategori}"
text2 = f"Fit: {pred_fitting} | {pred_warna}"
cv2.putText(image_display, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(image_display, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv2.imshow("Hasil (Enhanced)", image_display)
print("Tekan tombol apapun di jendela gambar untuk keluar...")
cv2.waitKey(0)
cv2.destroyAllWindows()