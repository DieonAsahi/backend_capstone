import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from color_detector import get_dominant_color_isolated, get_dominant_color_global
# Import fungsi enhancement (pastikan file image_enhancer.py sudah ada)
from image_enhancer import smart_enhance 

# --- Fungsi Helper ---
def load_and_prep_image(image_array, img_width, img_height):
    """Mempersiapkan gambar numpy untuk prediksi model."""
    img_display = image_array.copy()
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_width, img_height))
    
    # Preprocess khusus MobileNetV2
    img_processed = preprocess_input(img_resized) 
    
    img_batch = np.expand_dims(img_processed, axis=0)
    return img_batch, img_display

def get_prediction(model, img_batch, class_names):
    """Menjalankan prediksi pada model."""
    try:
        predictions = model.predict(img_batch, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = class_names[predicted_index]
        return predicted_class, confidence
    except Exception as e:
        print(f"Error prediksi: {e}")
        return "Unknown", 0.0

# --- FUNGSI UTAMA ---
def run_all_predictions(image_path, gender, models, rules, classes):
    """
    Alur: Smart Enhance -> Style -> Kategori -> Fitting (Logika User) -> Warna
    """
    
    # 1. Load Gambar
    img_original = cv2.imread(image_path)
    if img_original is None: 
        raise ValueError(f"Gagal memuat gambar: {image_path}")
    
    # 2. SMART ENHANCE (SOLUSI BAJU GELAP JADI HITAM)
    # Otomatis mencerahkan gambar jika gelap agar warna hijau/biru tua terlihat
    img_enhanced = smart_enhance(img_original)
    
    # 3. Siapkan batch gambar dari hasil enhance
    image_batch, _ = load_and_prep_image(img_enhanced, 224, 224)
    
    # 4. Prediksi STYLE
    pred_style, conf_style = get_prediction(models['style'], image_batch, classes['style'])
    
    # 5. Prediksi KATEGORI
    if gender == 'pria':
        model_kategori = models['men']
        class_kategori = classes['men']
    else: # wanita
        model_kategori = models['women']
        class_kategori = classes['women']

    pred_kategori, conf_kategori = get_prediction(model_kategori, image_batch, class_kategori)
    
    # --- 6. PREDIKSI FITTING (LOGIKA SESUAI REQUEST ANDA) ---
    pred_fitting = "-" 
    cat_lower = pred_kategori.lower() if pred_kategori else ""
    fit_key = None

    # List Sesuai Instruksi
    list_baju_wanita = ['blouse', 'outer', 'shirt', 'tshirt', 'blazer', 'dress', 'jacket']
    list_baju_pria = ['shirt', 'tshirt', 'outer', 'jacket', 'suit'] 
    
    try:
        # Logika Pria
        if gender == 'pria':
            if cat_lower in list_baju_pria: 
                fit_key = 'fit_pria_baju'
            elif cat_lower in ['pants']: 
                fit_key = 'fit_pria_celana'
        
        # Logika Wanita
        elif gender == 'wanita':
            if cat_lower in list_baju_wanita: 
                fit_key = 'fit_wanita_baju'
            elif cat_lower in ['pants']: 
                fit_key = 'fit_wanita_celana'
            elif cat_lower in ['skirt']: 
                fit_key = 'fit_wanita_rok'

        # Eksekusi Model Fitting
        if fit_key and fit_key in models['fit']:
            fit_res, _ = get_prediction(models['fit'][fit_key], image_batch, classes['fit'][fit_key])
            if fit_res: 
                pred_fitting = fit_res
                print(f"🎯 Fitting dideteksi: {pred_fitting} (menggunakan {fit_key})")
        else:
            pred_fitting = "-"

    except Exception as e:
        print(f"Error prediksi fitting: {e}")
        pred_fitting = "-"

    # 7. Prediksi WARNA (Gunakan img_enhanced agar akurat!)
    print("🎨 Mendeteksi warna pada gambar yang sudah dicerahkan...")
    pred_warna = get_dominant_color_isolated(img_enhanced)
    
    if not pred_warna or pred_warna in ['Error', 'Tidak terdeteksi']:
        pred_warna = get_dominant_color_global(img_enhanced)

    # 8. Return Hasil
    return {
        "style": pred_style,
        "category": pred_kategori,
        "fitting": pred_fitting, 
        "color": pred_warna
    }