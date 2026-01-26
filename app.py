import re
import io
import os
import base64
import uuid
import numpy as np
import threading 
import cv2
import csv
import random
import requests
import fal_client       
import pickle
import logging
import google.generativeai as genai
from config import Config # Import class Config
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from rembg import remove
from PIL import Image
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session, send_from_directory
from flask_mysqldb import MySQL
from flask_session import Session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from predict_skin import detect_skin_tone_ai
from image_enhancer import smart_enhance

# --- Variabel Global untuk Model ML ---
model_style, model_men, model_women = None, None, None

fit_models = {}
fit_classes = {}
AI_MODE_AKTIF = False
predict_logic = None 
prediction_lock = threading.Lock()

# --- Blok TRY...EXCEPT untuk memuat AI ---
try:
    import tensorflow as tf
    import predict_logic 
    import joblib
    
    # 1. Load Model Utama (.h5)
    STYLE_MODEL_PATH = 'model_style.h5'
    MEN_MODEL_PATH = 'model_kategori_pria.h5'
    WOMEN_MODEL_PATH = 'model_kategori_wanita.h5'
    
    print("Memuat model AI Utama (TensorFlow)...")
    model_style = tf.keras.models.load_model(STYLE_MODEL_PATH)
    model_men = tf.keras.models.load_model(MEN_MODEL_PATH)
    model_women = tf.keras.models.load_model(WOMEN_MODEL_PATH)

    # 2. Load Model Fitting (PENTING: Digunakan untuk fitur foto pakaian)
    print("⏳ Memuat Model Fitting...")
    FIT_CONFIGS = [
        ('fit_pria_baju',     'model_fit_pria_baju.h5',     ['Fit', 'Loose']), 
        ('fit_pria_celana',   'model_fit_pria_celana.h5',   ['Fit', 'Loose']),
        ('fit_wanita_baju',   'model_fit_wanita_baju.h5',   ['Fitted', 'Flare', 'Loose', 'Shoulder']), 
        ('fit_wanita_celana', 'model_fit_wanita_celana.h5', ['Fitted', 'Flare', 'Straight', 'Wide']),
        ('fit_wanita_rok',    'model_fit_wanita_rok.h5',    ['Flare', 'Mermaid', 'Straight'])
    ]

    for key, path, labels in FIT_CONFIGS:
        if os.path.exists(path):
            fit_models[key] = tf.keras.models.load_model(path)
            fit_classes[key] = labels
            print(f"   ✅ Loaded: {key}")
        else:
            print(f"   ❌ Missing: {key} (File tidak ditemukan)")
    
    AI_MODE_AKTIF = True
    print("✅ Model Visi (Foto Pakaian) berhasil dimuat.")

    # 3. Load Model Pakar Fashion (Random Forest .pkl)
    if os.path.exists('fashion_expert_model.pkl'):
        fashion_model = joblib.load('fashion_expert_model.pkl')
        fit_encoder = joblib.load('fit_encoder.pkl')
        print("✅ Fashion Expert ML Model (Rekomendasi) Loaded")
    else:
        fashion_model = None
        print("⚠️ Warning: File fashion_expert_model.pkl tidak ditemukan.")
    
except ImportError as e:
    print(f"⚠ PERINGATAN: Gagal impor library ML ({e}).")
except Exception as e:
    # Blok ini menangkap semua error lainnya (file korup, path salah, dll)
    fashion_model = None
    AI_MODE_AKTIF = False
    print(f"⚠ PERINGATAN: Terjadi kesalahan saat memuat model AI: {e}")

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config.from_object(Config) # Load semua config sekaligus
CORS(app)

# ==========================================
# 🔑 API KEY FAL.AI KAMU
os.environ['FAL_KEY'] = "ed5d330e-ae1e-4ebf-9756-1fb3efb67edb:2c1b9bc79e1801ed5d2a76d5cdd87bb3" 
# ==========================================

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not os.getenv("GEMINI_API_KEY"):
    print("❌ ERROR: GEMINI_API_KEY tidak ditemukan di file .env")
else:
    print("✅ GEMINI_API_KEY terdeteksi.")

google_client_id = os.getenv("GOOGLE_CLIENT_ID")
google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

if not google_client_id:
    print("❌ ERROR: GOOGLE_CLIENT_ID tidak ditemukan di file .env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_FILE = "rag_data.pkl"

# Load Otak AI hasil training dari model.py
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        rag_data = pickle.load(f)
    rag_df = rag_data["df"]
    rag_embeddings = rag_data["embeddings"]
    rag_model = SentenceTransformer(rag_data["model_name"])
    print("✅ RAG Data & SentenceTransformer Loaded")
else:
    print("⚠️ Warning: Jalankan model.py terlebih dahulu!")
    rag_df = None

# Gunakan model Flash 1.5 agar mendukung Vision (Analisis Gambar)
SELECTED_MODEL = "gemini-2.5-flash" 
SYSTEM_INSTRUCTION = (
    "Anda adalah StyleBot, seorang stylist pakar pakaian. "
    "Tugas utama anda HANYA memberikan cadangan atau analisis tentang outfit PAKAIAN. "
    "PERATURAN KETAT: "
    "1. Aplikasi ini bernama Stylo"
    "2. Nama anda adalah Mao, dan anda adalah stylo Bot"
    "3. Aplikasi ini adalah aplikasi rekomendasi pakaian dan lemari pakaian digital. "
    "4. Fokus hanya pada pakaian dalam jawaban anda." 
    "2. Berikan jawapan yang ringkas, santai dan dalam bahasa indonesia yang baik dan benar."
    "3. Jangan gunakan format markdown (seperti simbol bintang atau hashtag)."
    "7. Jika pengguna bertanya tentang aksesori (jam tangan, topi, beg), kaos kaki, atau topik selain pakaian dan 6 aturan lainnya, anda mesti menjawab: 'Maaf, saya hanya boleh membantu dengan pilihan baju.' "
)

gemini_model = genai.GenerativeModel(
    model_name=SELECTED_MODEL,
    system_instruction=SYSTEM_INSTRUCTION
)

# --- Konfigurasi MySQL ---
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_swipeer'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['MYSQL_AUTOCOMMIT'] = True # Tambahkan baris ini

# --- Konfigurasi Session ---
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Inisialisasi MySQL Extension
mysql = MySQL(app)

# Helper kursor yang bener buat library lu (Tanpa parameter aneh)
def get_db_cursor():
    try:
        mysql.connection.ping()
    except:
        pass
    return mysql.connection.cursor()
# --- Direktori Upload ---
UPLOAD_DIR = os.path.join(app.root_path, 'static/uploads')
UPLOAD_FOLDER = 'static/uploads'
TEMP_FOLDER = 'static/temp'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 1. LOGIKA ATURAN VALIDASI ---
VALID_RULES = {
    'pria': { 'Formal': ['suit', 'shirt', 'pants', 'tshirt'], 'Casual': ['outer', 'pants', 'shirt', 'tshirt'], 'Sport': ['jacket', 'pants', 'tshirt'] },
    'wanita': { 'Formal': ['blazer', 'blouse', 'dress', 'pants', 'shirt', 'skirt', 'tshirt', 'outer'], 'Casual': ['blouse', 'outer', 'pants', 'shirt', 'skirt', 'tshirt'], 'Sport': ['jacket', 'pants', 'tshirt', 'skirt'] }
}
MODEL_CLASSES = {
    'style': ['Sport', 'Casual', 'Formal'],
    'men': ['jacket', 'outer', 'pants', 'shirt', 'suit', 'tshirt'], 
    'women': ['jacket', 'blazer', 'blouse', 'dress', 'outer', 'pants', 'shirt', 'skirt', 'tshirt'],
    'fit': fit_classes 
}

# --- 2. RUTE AUTENTIKASI ---

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify(success=False, message="Username & password wajib diisi"), 400

    try:
        cur = mysql.connection.cursor()
        
        # Ambil data user berdasarkan username
        cur.execute("""
            SELECT user_id, name, username, email, password_hash, gender, profile_photo_url 
            FROM users WHERE username=%s
        """, [username])
        
        user = cur.fetchone()
        cur.close()

        # PERBAIKAN: Gunakan key string ['password_hash'], bukan index angka [4]
        if user and check_password_hash(user['password_hash'], password):
            return jsonify({
                "success": True, 
                "message": "Login Sukses", 
                "user_id": user['user_id'], 
                "name": user['name'],
                "username": user['username'], 
                "email": user['email'], 
                "gender": user['gender'], 
                "photo_url": user['profile_photo_url']
            })
        
        return jsonify(success=False, message="Pastikan username dan password benar"), 401

    except Exception as e:
        print(f"Error Login: {e}") # Print error di terminal agar terlihat
        return jsonify(success=False, message=f"Error Server: {str(e)}"), 500
    
@app.route('/google_login', methods=['POST'])
def google_login():
    try:
        data = request.json
        email = data.get('email')
        google_name = data.get('name') # Ambil Display Name dari Google
        google_username = data.get('username') 
        photo_url = data.get('photo_url')

        cur = mysql.connection.cursor()
        # Cari user berdasarkan email
        cur.execute("SELECT user_id, name, username, email FROM users WHERE email = %s", [email])
        user = cur.fetchone()

        if not user:
            # Jika user belum ada, daftarkan dengan google_name sebagai 'name'
            cur.execute(
                "INSERT INTO users (name, username, email, password_hash, profile_photo_url) VALUES (%s, %s, %s, %s, %s)", 
                (google_name, google_username, email, 'GOOGLE_AUTH', photo_url)
            )
            mysql.connection.commit()
            user_id = cur.lastrowid
            cur.execute("SELECT user_id, name, username, email FROM users WHERE user_id = %s", [user_id])
            user = cur.fetchone()
        
        cur.close()

        # Kembalikan field 'name' yang benar dari database
        return jsonify({
            "success": True,
            "user_id": user['user_id'],
            "name": user['name'], # Ini yang akan ditampilkan di Flutter
            "username": user['username'],
            "email": user['email'],
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
            
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    
    # Ambil data sesuai input dari Flutter
    name = data.get('name')         # Masuk ke kolom 'name' baru
    username = data.get('username') # Masuk ke kolom 'username'
    password = data.get('password')

    # Validasi input (Email tidak wajib di sini)
    if not all([name, username, password]):
        return jsonify(success=False, message="Data tidak lengkap"), 400

    hashed = generate_password_hash(password)

    try:
        cur = mysql.connection.cursor()
        
        # Cek apakah USERNAME sudah terdaftar (sebelumnya cek email)
        cur.execute("SELECT user_id FROM users WHERE username=%s", [username])
        if cur.fetchone():
            cur.close()
            return jsonify(success=False, message="Username sudah digunakan"), 409

        # Insert data: name, username, password_hash
        # Kolom email tidak dimasukkan (akan otomatis NULL)
        cur.execute(
            "INSERT INTO users (name, username, password_hash) VALUES (%s, %s, %s)",
            (name, username, hashed)
        )
        
        mysql.connection.commit()
        cur.close()

        return jsonify(success=True, message="Registrasi berhasil")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500
    
@app.route('/api/user/<int:user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    try:
        cur = mysql.connection.cursor()
        # PASTIIN 'bio' ada di SELECT ini
        cur.execute("""
            SELECT user_id, name, username, email, gender, profile_photo_url, bio 
            FROM users WHERE user_id = %s
        """, [user_id])
        user = cur.fetchone()
        cur.close()
        
        if user:
            return jsonify({
                "success": True,
                "user_id": user['user_id'],
                "name": user['name'],
                "username": user['username'],
                "email": user['email'],
                "gender": user['gender'],
                "photo_url": user['profile_photo_url'],
                "bio": user['bio'] # BARIS INI WAJIB ADA
            }), 200
        else:
            return jsonify({"success": False, "message": "User not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# BIAR GENDER BISA DIUBAH (HAPUS PENGECEKAN KAKU)
@app.route('/api/user/update_gender', methods=['POST'])
def update_gender_only():
    data = request.get_json()
    user_id = data.get('user_id')
    gender_ui = data.get('gender')
    
    gender_db = "male" if gender_ui == "Laki-laki" else "female"

    try:
        cur = mysql.connection.cursor()
        # Hapus pengecekan 'if row['gender']' supaya lu bisa update kapan aja
        cur.execute("UPDATE users SET gender=%s WHERE user_id=%s", (gender_db, user_id))
        mysql.connection.commit()
        cur.close()
        return jsonify(success=True, message="Gender berhasil diperbarui")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500    
    
@app.route('/api/user/update', methods=['POST'])
def update_user_profile():
    # 1. Ambil Data (Support JSON & Form-Data)
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    user_id = data.get('user_id')
    username = data.get('username')
    name = data.get('name') # <--- TAMBAHAN 1: Ambil data nama
    gender_ui = data.get('gender') 
    bio = data.get('bio')

    if not user_id:
        return jsonify(success=False, message="User ID tidak ditemukan"), 400

    cur = mysql.connection.cursor()

    # ---------------------------------------------------------
    # 2. VALIDASI USERNAME
    # ---------------------------------------------------------
    if username:
        cur.execute("SELECT user_id FROM users WHERE username=%s AND user_id != %s", (username, user_id))
        existing_user = cur.fetchone()
        if existing_user:
            cur.close()
            return jsonify(success=False, message="Username sudah digunakan, silakan pilih yang lain."), 409

    # ---------------------------------------------------------
    # 3. LOGIKA UPLOAD FOTO
    # ---------------------------------------------------------
    photo_db_url = None
    if 'profile_image' in request.files:
        file = request.files['profile_image']
        if file and file.filename != '':
            try:
                filename = f"{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(UPLOAD_DIR, filename)
                file.save(file_path)
                photo_db_url = f"static/uploads/{filename}"
            except Exception as e:
                print(f"Gagal simpan gambar: {e}")

    # ---------------------------------------------------------
    # 4. LOGIKA GENDER
    # ---------------------------------------------------------
    gender_db = None
    if gender_ui:
        if gender_ui.lower() == "laki-laki":
            gender_db = "male"
        elif gender_ui.lower() == "perempuan":
            gender_db = "female"

    # ---------------------------------------------------------
    # 5. BANGUN QUERY DINAMIS
    # ---------------------------------------------------------
    update_query = "UPDATE users SET "
    params = []

    if username:
        update_query += "username=%s, "
        params.append(username)
    
    # <--- TAMBAHAN 2: Masukkan nama ke database
    if name:
        update_query += "name=%s, "
        params.append(name)
    
    if gender_db:
        update_query += "gender=%s, "
        params.append(gender_db)
        
    if photo_db_url:
        update_query += "profile_photo_url=%s, "
        params.append(photo_db_url)

    if bio is not None:
        update_query += "bio=%s, "
        params.append(bio)

    if not params:
        cur.close()
        return jsonify(success=True, message="Tidak ada perubahan data")

    update_query = update_query.rstrip(', ') + " WHERE user_id=%s"
    params.append(user_id)

    try:
        cur.execute(update_query, tuple(params))
        mysql.connection.commit()

        # Ambil data terbaru
        cur.execute("SELECT profile_photo_url, bio, name FROM users WHERE user_id=%s", [user_id])
        updated_user = cur.fetchone()
        cur.close()

        return jsonify(
            success=True, 
            message="Profil berhasil diperbarui", 
            new_photo_url=updated_user['profile_photo_url'] if updated_user else None,
            new_bio=updated_user['bio'] if updated_user else None,
            new_name=updated_user['name'] if updated_user else None # <--- TAMBAHAN 3: Kembalikan nama baru
        )

    except Exception as e:
        cur.close()
        print(f"Error Update DB: {e}")
        return jsonify(success=False, message=str(e)), 500
                
def cari_dataset(pesan_user):
    if rag_df is None: return None
    user_embedding = rag_model.encode(pesan_user, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, rag_embeddings)[0]
    best_idx = np.argmax(cos_scores.cpu().numpy())
    if cos_scores[best_idx].item() > 0.5:
        return rag_df.iloc[best_idx]["jawaban"]
    return None

def tentukan_motion(teks):
    teks = teks.lower()
    if any(k in teks for k in ["keren", "bagus", "cocok", "hai"]): return "TapBody"
    if any(k in teks for k in ["kurang", "pikir", "bingung"]): return "Surprise"
    return "Idle"

@app.route("/view")
def view_live2d():
    return render_template('live2d.html')

@app.route('/static/model/<path:filename>')
def serve_model(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/model'), filename)

@app.route("/get_response", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    referensi = cari_dataset(user_msg)
    prompt = f"Referensi: {referensi}. Pertanyaan: {user_msg}" if referensi else user_msg
    try:
        response = gemini_model.generate_content(prompt)
        jawaban = response.text.replace("*", "")
        return jsonify({"response": jawaban, "motion": tentukan_motion(jawaban)})
    except:
        return jsonify({"response": "Maaf, coba lagi nanti."}), 500

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    data = request.get_json()
    image_data = base64.b64decode(data["image"])
    
    try:
        prompt_vision = (
            "Perhatikan gambar ini. Jika ini adalah gambar pakaian/celana/outfit fashion, "
            "berikan komentar singkat, ceria, dan memuji (maksimal 2 kalimat) tentang gaya tersebut. "
            "Jika gambar ini BUKAN pakaian (misalnya hanya wajah, pemandangan, atau benda acak), "
            "katakan: 'Maaf, Mao hanya bisa menilai outfit pakaian kamu ya!'."
        )

        response = gemini_model.generate_content([
            prompt_vision,
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        jawaban = response.text.replace("*", "").strip()
        return jsonify({
            "full_text": jawaban, 
            "motion": tentukan_motion(jawaban) 
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Gagal analisis gambar"}), 500
        
# --- WARDROBE ACTIONS ---
@app.route('/api/wardrobe/item', methods=['POST'])
def wardrobe_item():
    user_id = session.get("user_id") or request.json.get("user_id")
    if not user_id: return jsonify(error="Not logged in"), 401

    data = request.get_json()
    action = data.get('action') 

    if action == 'predict':
        image_data_url = data.get('image_data')

        cur = mysql.connection.cursor()
        cur.execute("SELECT gender FROM users WHERE user_id=%s", [user_id])
        row = cur.fetchone()
        cur.close()

        if not row or not row['gender']:
            return jsonify(error="isi gender kamu ya"), 400

        gender = row['gender'].lower()
        if gender in ['male', 'laki-laki', 'laki', 'm']: gender = 'pria'
        elif gender in ['female', 'perempuan', 'wanita', 'f']: gender = 'wanita'
        else: return jsonify(error=f"Gender tidak valid: {row['gender']}"), 400

        header, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)

        nparr = np.frombuffer(image_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. TERAPKAN SMART ENHANCE (Gamma Correction) DI SINI
        # Agar AI melihat gambar yang terang/jelas
        img_enhanced = smart_enhance(img_cv)

        temp_path = os.path.join(UPLOAD_DIR, f"temp_{user_id}.jpg")
        with open(temp_path, "wb") as f:
            f.write(image_data)

        models = {
            'style': model_style, 
            'men': model_men, 
            'women': model_women,
            'fit': fit_models 
        }

        with prediction_lock: 
            prediction = predict_logic.run_all_predictions(
                temp_path, gender, models, VALID_RULES, MODEL_CLASSES
            )

        if os.path.exists(temp_path): os.remove(temp_path)

        return jsonify(
            success=True,
            prediction=prediction,
            valid_categories=VALID_RULES.get(gender, {}).get(prediction['style'], [])
        )

    if action == 'save':
        style = data.get('style')
        parent_name = data.get('category')
        child_name = data.get('fitting')
        color_name = data.get('color')
        image_data_url = data.get('image_data')
        gender = data.get('gender')

        if not all([style, parent_name, color_name, image_data_url, gender]):
            return jsonify(error="Data tidak lengkap"), 400

        style = style.strip().title()
        gender = gender.strip().lower()
        parent_name = parent_name.strip().lower()
        
        if not child_name: 
            child_name = "Standard"
        else:
            child_name = child_name.strip()

        color_name = normalize_color_name(color_name)

        if gender in ['male', 'laki-laki', 'm']: gender = 'pria'
        elif gender in ['female', 'perempuan', 'f']: gender = 'wanita'

        if gender not in VALID_RULES or style not in VALID_RULES[gender] or parent_name not in VALID_RULES[gender][style]:
            return jsonify(error=f"Validasi Kategori Gagal: {style} - {parent_name}"), 400

        header, encoded = image_data_url.split(",", 1)
        image_data_bytes = base64.b64decode(encoded)
        
        try:
            nparr = np.frombuffer(image_data_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_enhanced = smart_enhance(img_cv)
            img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(img_rgb)
            output_image = remove(input_image) 
            final_img = Image.new("RGB", output_image.size, (255, 255, 255))
            if len(output_image.split()) == 4:
                final_img.paste(output_image, mask=output_image.split()[3])
            else:
                final_img = output_image.convert("RGB")

            filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join(UPLOAD_DIR, filename)
            final_img.save(image_path, "JPEG", quality=90)
            
        except Exception as e:
            print(f"Gagal memproses gambar (Enhance/Rembg): {e}")
            filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join(UPLOAD_DIR, filename)
            with open(image_path, "wb") as f:
                f.write(image_data_bytes)

        image_db_path = f"static/uploads/{filename}"

        cur = mysql.connection.cursor()

        cur.execute("SELECT category_id FROM categories WHERE category_name=%s AND parent_category_id IS NULL", [parent_name])
        p_row = cur.fetchone()
        if p_row:
            parent_id = p_row['category_id']
        else:
            cur.execute("INSERT INTO categories (category_name, parent_category_id) VALUES (%s, NULL)", [parent_name])
            mysql.connection.commit()
            parent_id = cur.lastrowid
            
        cur.execute("SELECT category_id FROM categories WHERE category_name=%s AND parent_category_id=%s", (child_name, parent_id))
        c_row = cur.fetchone()
        if c_row:
            final_category_id = c_row['category_id']
        else:
            cur.execute("INSERT INTO categories (category_name, parent_category_id) VALUES (%s, %s)", (child_name, parent_id))
            mysql.connection.commit()
            final_category_id = cur.lastrowid

        cur.execute("SELECT color_id FROM colors WHERE color_name=%s", [color_name])
        clr = cur.fetchone()
        if not clr:
            cur.execute("INSERT INTO colors (color_name) VALUES (%s)", [color_name])
            mysql.connection.commit()
            color_id = cur.lastrowid
        else:
            color_id = clr['color_id']

        item_full_name = f"{style} {child_name} {parent_name} {color_name}"
        cur.execute("""
            INSERT INTO user_wardrobe (user_id, item_name, image_url, style, category_id, color_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, item_full_name, image_db_path, style, final_category_id, color_id))

        mysql.connection.commit()
        cur.close()

        return jsonify(success=True, message="Pakaian berhasil disimpan dengan detail fitting!")

    return jsonify(error="Invalid action"), 400

def normalize_color_name(color):
    if not color: return None
    return color.strip().title()

@app.route('/api/wardrobe/<int:user_id>', methods=['GET'])
def api_get_wardrobe(user_id):
    style = request.args.get('style') 
    try:
        cur = mysql.connection.cursor()
        
        # REVISI: Tambahkan "AND uw.type = 'owned'"
        # Agar yang muncul di lemari HANYA baju yang dimiliki, bukan wishlist/favorit.
        query_items = """
            SELECT 
                uw.item_id, uw.item_name, uw.image_url, uw.style,
                clr.color_name,
                c_child.category_name as child_name,
                c_parent.category_name as parent_name
            FROM user_wardrobe uw
            LEFT JOIN categories c_child ON uw.category_id = c_child.category_id
            LEFT JOIN categories c_parent ON c_child.parent_category_id = c_parent.category_id
            LEFT JOIN colors clr ON uw.color_id = clr.color_id
            WHERE uw.user_id = %s AND uw.type = 'owned' 
        """
        
        params = [user_id]
        
        if style and style.lower() != "semua":
            query_items += " AND LOWER(uw.style) = LOWER(%s)"
            params.append(style)
            
        query_items += " ORDER BY uw.item_id DESC"
        
        cur.execute(query_items, params)
        raw_items = cur.fetchall()
        
        processed_items = []
        unique_categories = set()
        
        for item in raw_items:
            real_category = item['parent_name'] if item['parent_name'] else item['child_name']
            
            processed_items.append({
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'image_url': item['image_url'],
                'style': item['style'],
                'color_name': item['color_name'],
                'category_name': real_category 
            })
            
            if real_category:
                unique_categories.add(real_category)
        
        categories_list = sorted(list(unique_categories))
        
        cur.close()
        return jsonify(success=True, items=processed_items, categories=categories_list)
        
    except Exception as e:
        print(f"Error Get Wardrobe: {e}")
        return jsonify(success=False, message=str(e)), 500
        
def classify_body_shape(bust, waist, hip):
    ratio_bust_hip = bust / hip
    ratio_waist_bust = waist / bust
    ratio_waist_hip = waist / hip
    if abs(bust - hip) / hip <= 0.05 and ratio_waist_bust <= 0.75 and ratio_waist_hip <= 0.75: return 'Hourglass'
    if hip >= bust * 1.05: return 'Pear'
    if bust >= hip * 1.05: return 'Inverted Triangle'
    if ratio_waist_bust >= 0.75 and ratio_waist_hip >= 0.75 and abs(bust - hip) / hip <= 0.05: return 'Rectangle'
    if waist >= bust * 0.85 and bust > hip: return 'Apple'
    return 'Unknown'

@app.route('/api/bodyshape/calculate', methods=['POST'])
def calculate_bodyshape():
    data = request.json
    try:
        bust = float(data['bust'])
        waist = float(data['waist'])
        hip = float(data['hip'])
        body_shape = classify_body_shape(bust, waist, hip)
        
        descriptions = {
            "Hourglass": "Bahu dan pinggul seimbang dengan pinggang ramping.",
            "Pear": "Pinggul lebih besar dari bahu. Cocok fokus ke atasan.",
            "Inverted Triangle": "Bahu lebih lebar dari pinggul.",
            "Rectangle": "Bahu, pinggang, dan pinggul hampir sejajar.",
            "Apple": "Bagian tengah tubuh lebih dominan.",
            "Unknown": "Ukuran tidak dapat diklasifikasikan."
        }
        
        return jsonify({
            "success": True, 
            "body_shape": body_shape,
            "description": descriptions.get(body_shape, "Deskripsi tidak tersedia.") 
        })
    except Exception as e:
        return jsonify(success=False, message=str(e)), 400
    
BODY_SHAPE_MAP = {"Hourglass": 1, "Pear": 2, "Inverted Triangle": 3, "Rectangle": 4, "Apple": 5}

@app.route('/api/bodyshape/save', methods=['POST'])
def save_bodyshape():
    data = request.json
    user_id = data.get('user_id')
    shape_name = data.get('body_shape')
    body_shape_id = BODY_SHAPE_MAP.get(shape_name)
    if not body_shape_id: return jsonify({"success": False}), 400
    
    cur = mysql.connection.cursor()
    cur.execute("UPDATE users SET body_shape_id = %s WHERE user_id = %s", (body_shape_id, user_id))
    mysql.connection.commit()
    cur.close()
    return jsonify({"success": True})

@app.route('/api/bodyshape/<int:user_id>', methods=['GET'])
def get_bodyshape(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT bs.shape_name FROM users u LEFT JOIN body_shapes bs ON u.body_shape_id = bs.body_shape_id WHERE u.user_id = %s", [user_id])
    row = cur.fetchone()
    cur.close()
    return jsonify({"success": True, "body_shape": row['shape_name'] if row else None})

@app.route('/scan_face', methods=['POST'])
def scan_face():
    data = request.get_json()
    user_id = data.get("user_id")
    confirm = data.get("confirm", False)
    
    if not user_id: return jsonify(error="Unauthorized"), 401

    # --- JIKA CONFIRM (SIMPAN DATA) ---
    if confirm:
        skin_tone = data.get("skin_tone")
        cur = mysql.connection.cursor()
        cur.execute("SELECT skin_tone_id FROM skin_tones WHERE tone_name=%s", [skin_tone])
        row = cur.fetchone()
        if not row:
            cur.execute("INSERT INTO skin_tones (tone_name) VALUES (%s)", [skin_tone])
            mysql.connection.commit()
            skin_tone_id = cur.lastrowid
        else:
            skin_tone_id = row["skin_tone_id"]
        
        cur.execute("UPDATE users SET skin_tone_id=%s WHERE user_id=%s", (skin_tone_id, user_id))
        mysql.connection.commit()
        cur.close()
        return jsonify(status="saved", skin_tone=skin_tone)

    # --- PROSES ANALISIS GAMBAR ---
    try:
        img_url = data.get("image")
        header, encoded = img_url.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ==========================================
        # 1. VALIDASI CAHAYA (BRIGHTNESS)
        # ==========================================
        # Ubah ke HSV untuk mengambil nilai Value (Kecerahan)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2]) # Ambil rata-rata channel V
        
        print(f"📸 Brightness Level: {brightness}")

        # Ambang batas (Threshold) - Sesuaikan jika perlu
        if brightness < 50:
            return jsonify(status="error", message="Foto terlalu gelap. Cari tempat yang lebih terang."), 400
        
        if brightness > 210:
            return jsonify(status="error", message="Foto terlalu terang. Hindari cahaya lampu langsung."), 400

        # ==========================================
        # 2. SIMPAN & DETEKSI WAJAH (HAAR CASCADE)
        # ==========================================
        filename = f"preview_{user_id}.jpg"
        path = os.path.join(UPLOAD_DIR, filename)
        cv2.imwrite(path, img)
        
        # Panggil fungsi dari predict_skin.py
        ai_result = detect_skin_tone_ai(path)
        print(f"🤖 AI Output: {ai_result}")

        # ==========================================
        # 3. HANDLING ERROR DARI AI
        # ==========================================
        
        # Mapping pesan error dari predict_skin.py ke pesan user
        if ai_result == "Wajah Tidak Terdeteksi":
            return jsonify(status="error", message="Area tidak teridentifikasi, harap scan wajah Anda."), 400
        
        if ai_result == "ROI Error" or ai_result == "Gambar Tidak Valid":
            return jsonify(status="error", message="Posisi tidak pas, harap scan wajah Anda dengan benar."), 400
            
        if ai_result == "Model Tidak Tersedia":
            return jsonify(status="error", message="Server Error: Model belum dimuat."), 500

        return jsonify(status="preview", skin_tone=ai_result)

    except Exception as e:
        print(f"❌ Error Scan Face: {e}")
        return jsonify(status="error", message=str(e)), 500
    
OUTFIT_COMBINATIONS = {
    'pria': {
        'formal': [['suit', 'shirt', 'pants'], ['shirt', 'pants'], ['tshirt', 'pants']],
        'casual': [['outer', 'tshirt', 'pants'], ['shirt', 'pants'], ['tshirt', 'pants']],
        'sport': [['jacket', 'tshirt', 'pants'], ['tshirt', 'pants']]
    },
    'wanita': {
        'formal': [
            ['dress'], 
            ['blazer', 'pants'], ['blazer', 'skirt'], 
            ['blouse', 'pants'], ['blouse', 'skirt'],
            ['shirt', 'pants'], ['shirt', 'skirt'],
            ['outer', 'shirt', 'pants']
        ],
        'casual': [
            ['blouse', 'pants'], ['blouse', 'skirt'],
            ['outer', 'tshirt', 'pants'], 
            ['shirt', 'pants'], 
            ['tshirt', 'skirt'], ['tshirt', 'pants']
        ],
        'sport': [
            ['jacket', 'tshirt', 'pants'], 
            ['tshirt', 'pants'], 
            ['tshirt', 'skirt'] 
        ]
    }
}

def get_apriori_weights(user_id, available_combos):
    """
    Belajar dari tabel outfits & outfit_items.
    Menghitung kategori mana yang paling sering disimpan user.
    """
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT c.category_name, COUNT(*) as frekuensi
        FROM outfit_items oi
        JOIN outfits o ON oi.outfit_id = o.outfit_id
        JOIN user_wardrobe uw ON oi.item_id = uw.item_id
        JOIN categories c ON uw.category_id = c.category_id
        WHERE o.user_id = %s
        GROUP BY c.category_name
    """, [user_id])
    rows = cur.fetchall()
    cur.close()

    freq_map = {row['category_name'].lower(): row['frekuensi'] for row in rows}
    
    weighted_combos = []
    weights = []
    
    for combo in available_combos:
        score = 1 
        for item in combo:
            score += freq_map.get(item, 0) * 10 
        
        weighted_combos.append(combo)
        weights.append(score)
        
    return weighted_combos, weights

@app.route('/api/recommendation/visualize', methods=['POST'])
def visualize_recommendation():
    try:
        user_id = request.form.get('user_id')
        
        top_id = request.form.get('top_id') 
        bottom_id = request.form.get('bottom_id')
        
        top_url_online = request.form.get('top_image_url')
        bottom_url_online = request.form.get('bottom_image_url')

        person_image = request.files.get('person_image')

        if not person_image or not user_id:
            return jsonify({"status": "error", "message": "Foto badan wajib ada!"}), 400

        temp_person_path = os.path.join(TEMP_FOLDER, f"vis_user_{user_id}.jpg")
        person_image.save(temp_person_path)

        current_processing_image = temp_person_path 
        cur = mysql.connection.cursor()
        final_url = ""
        processed_count = 0

        def get_garment_info(local_id, online_url, type_name):
            if online_url and online_url != 'null':
                ext = "jpg"
                temp_filename = f"temp_online_{type_name}_{uuid.uuid4()}.{ext}"
                save_path = os.path.join(TEMP_FOLDER, temp_filename)
                if download_image_from_url(online_url, save_path):
                    return save_path, "Online Item"
            elif local_id and local_id != 'null':
                cur.execute("SELECT image_url, item_name, c.color_name FROM user_wardrobe uw LEFT JOIN colors c ON uw.color_id = c.color_id WHERE uw.item_id = %s", [local_id])
                row = cur.fetchone()
                if row:
                    full_path = os.path.join(app.root_path, row['image_url'])
                    desc = f"{row['color_name'] or ''} {row['item_name']}"
                    return full_path, desc
            return None, None

        garment_path, desc = get_garment_info(top_id, top_url_online, "top")
        if garment_path:
            try:
                print(f"👕 Processing Top: {desc}")
                final_url = call_fal_api(current_processing_image, garment_path, "atas", desc)
                
                step1_path = os.path.join(TEMP_FOLDER, f"vis_step1_{user_id}.jpg")
                if download_image_from_url(final_url, step1_path):
                    current_processing_image = step1_path 
                    processed_count += 1
            except Exception as e:
                print(f"⚠️ Gagal atasan: {e}")

        # 2. BAWAHAN
        garment_path, desc = get_garment_info(bottom_id, bottom_url_online, "bottom")
        if garment_path:
            try:
                print(f"👖 Processing Bottom: {desc}")
                final_url = call_fal_api(current_processing_image, garment_path, "bawah", desc)
                processed_count += 1
            except Exception as e:
                print(f"⚠️ Gagal bawahan: {e}")

        cur.close()
        
        if processed_count == 0: 
            return jsonify({"status": "error", "message": "Gagal memproses gambar"}), 500
            
        return jsonify({"status": "success", "result_url": final_url})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
                    
@app.route("/api/recommendation/final", methods=["POST"])
def final_recommendation():
    data = request.get_json()
    user_id = data.get("user_id")
    style = data.get("style", "casual").lower()
    gender = data.get("gender", "pria").lower()
    source = data.get("source", "lemari")

    if not user_id: return jsonify(success=False, message="User ID error"), 400

    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT skin_tone_id, body_shape_id FROM users WHERE user_id=%s", [user_id])
        user_row = cur.fetchone()
        
        if not user_row or not user_row['skin_tone_id'] or not user_row['body_shape_id']:
            cur.close()
            return jsonify(success=False, message="Lengkapi profil warna kulit dan bentuk tubuh Anda."), 400
            
        skin_tone_id = user_row['skin_tone_id']
        body_shape_id = user_row['body_shape_id']
        
        gender_combos = OUTFIT_COMBINATIONS.get(gender, {})
        style_combos = gender_combos.get(style, [["tshirt", "pants"]])
        combos, weights = get_apriori_weights(user_id, style_combos)
        chosen_structure = random.choices(combos, weights=weights, k=1)[0]
        
        if 'dress' in chosen_structure: chosen_structure = ['dress']

        final_recommendations = []
        total_ml_score = 0

        for category_keyword in chosen_structure:
            if source == 'online':
                query = """
                    SELECT oc.catalog_item_id as item_id, oc.item_name, oc.image_url, 
                           oc.category_id, oc.color_id, c_child.category_name as fitting_name
                    FROM online_catalog oc
                    LEFT JOIN categories c_child ON oc.category_id = c_child.category_id
                    LEFT JOIN categories c_parent ON c_child.parent_category_id = c_parent.category_id
                    WHERE LOWER(oc.style)=%s AND (LOWER(c_parent.category_name)=%s OR LOWER(c_child.category_name)=%s)
                """
                params = [style, category_keyword, category_keyword]
            else:
                query = """
                    SELECT uw.item_id, uw.item_name, uw.image_url, uw.category_id, uw.color_id,
                           c_child.category_name as fitting_name
                    FROM user_wardrobe uw
                    LEFT JOIN categories c_child ON uw.category_id = c_child.category_id
                    LEFT JOIN categories c_parent ON c_child.parent_category_id = c_parent.category_id
                    WHERE uw.user_id=%s AND uw.type = 'owned' AND LOWER(uw.style)=%s
                    AND (LOWER(c_parent.category_name)=%s OR LOWER(c_child.category_name)=%s)
                """
                params = [user_id, style, category_keyword, category_keyword]

            cur.execute(query, params)
            items = cur.fetchall()
            if not items: continue

            scored_items = []
            for item in items:
                if fashion_model:
                    try:
                        f_name = item['fitting_name'] if item['fitting_name'] else "Fit"
                        fit_encoded = fit_encoder.transform([f_name])[0]
                        
                        features = [[skin_tone_id, body_shape_id, item['color_id'], fit_encoded, item['category_id']]]
                        
                        prob = fashion_model.predict_proba(features)[0][1]
                        match_score = int(prob * 100)
                    except:
                        match_score = 70 
                else:
                    match_score = 70

                cur.execute("SELECT notes FROM recommendation_rules WHERE skin_tone_id=%s AND color_id=%s LIMIT 1", 
                            (skin_tone_id, item['color_id']))
                rule_note = cur.fetchone()
                
                scored_items.append({
                    **item, 
                    "match_score": match_score,
                    "notes": rule_note['notes'] if rule_note else "Warna ini sangat stabil untuk gayamu."
                })

            if scored_items:
                best_item = max(scored_items, key=lambda x: x['match_score'])
                final_recommendations.append(best_item)
                total_ml_score += best_item['match_score']

        avg_score = round(total_ml_score / len(final_recommendations)) if final_recommendations else 0
        
        cur.close()
        return jsonify({
            "success": True, 
            "recommendations": final_recommendations, 
            "total_match_score": avg_score, 
            "summary": f"Berdasarkan analisis mao, outfit ini {avg_score}% cocok dengan profil tubuh dan warna kulitmu.", 
            "structure": chosen_structure
        })
        
    except Exception as e:
        if cur: cur.close()
        print(f"Error Recommendation: {e}")
        return jsonify(success=False, message=str(e)), 500
            
@app.route("/api/outfit/save", methods=["POST"])
def save_outfit():
    data = request.get_json()
    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO outfits (user_id, outfit_name, bot_feedback_text, bot_feedback_score) VALUES (%s, %s, %s, %s)",
                    (data['user_id'], data['outfit_name'], data.get('feedback_text'), data.get('feedback_score')))
        oid = cur.lastrowid
        for iid in data['item_ids']:
            cur.execute("INSERT INTO outfit_items (outfit_id, item_id) VALUES (%s, %s)", (oid, iid))
        mysql.connection.commit()
        cur.close()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

def resize_and_convert_to_base64(image_path, max_size=1536):
    """
    Membaca gambar, mengecilkan ukuran sedikit (agar tidak timeout), 
    tapi menjaga kualitas tetap tajam (HD).
    """
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            width, height = img.size
            
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"⬇️ Resize gambar (HD): {width}x{height} -> {new_size}")
            else:
                print(f"✨ Gambar sudah ukuran optimal: {width}x{height}")
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{data}"
    except Exception as e:
        print(f"❌ Error resize: {e}")
        return None
        
def download_image_from_url(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"❌ Gagal download: {e}")
    return False

def call_fal_api(human_img_path, garment_img_path, mode, item_desc="clothing item"):
    print(f"🚀 [{datetime.now().strftime('%H:%M:%S')}] AI Process ({mode}) Using IDM-VTON")

    human_base64 = resize_and_convert_to_base64(human_img_path)
    garment_base64 = resize_and_convert_to_base64(garment_img_path)

    if not human_base64 or not garment_base64:
        raise Exception("Gagal memproses gambar (resize error)")

    category_value = "upper_body" 
    final_desc = ""
    desc_lower = item_desc.lower()
    
    if mode == "atas":
        if "dress" in desc_lower or "gaun" in desc_lower:
             category_value = "dresses"
             final_desc = f"model wearing {item_desc}, dress, one piece garment, full body"
        else:
             category_value = "upper_body"
             
             prompt_suffix = ""
             if any(k in desc_lower for k in ["tshirt", "kaos", "short", "pendek", "polo"]):
                 prompt_suffix = ", short sleeves, arms visible"
             elif any(k in desc_lower for k in ["shirt", "kemeja", "jacket", "hoodie", "long", "panjang"]):
                 prompt_suffix = ", long sleeves"
             
             final_desc = f"model wearing {item_desc}, upper body garment{prompt_suffix}"
             
    else:
        category_value = "lower_body"
        final_desc = f"model wearing {item_desc}, on legs, pants, trousers, skirt, lower body garment"

    print(f"👉 Params: Cat={category_value} | Desc={final_desc}")

    try:
        handler = fal_client.submit(
            "fal-ai/idm-vton",
            arguments={
                "human_image_url": human_base64,
                "garment_image_url": garment_base64,
                "category": category_value,
                "description": final_desc,
                "seed": 42 
            }
        )
        result = handler.get()
        print(f"✅ Selesai AI ({mode}).")
        return result['image']['url']
        
    except Exception as e:
        print(f"❌ Error Fal.ai: {e}")
        raise e
            
@app.route('/api/process-outfit', methods=['POST'])
def process_outfit():
    cur = mysql.connection.cursor()
    try:
        user_id = request.form.get('user_id')
        top_item_id = request.form.get('top_item_id')
        bottom_item_id = request.form.get('bottom_item_id')
        person_image = request.files['person_image']

        temp_person_path = os.path.join(TEMP_FOLDER, f"temp_user_{user_id}.jpg")
        person_image.save(temp_person_path)
        
        current_path = temp_person_path 
        final_url = ""

        if top_item_id and top_item_id != 'null':
            cur.execute("SELECT image_url, item_name, c.color_name FROM user_wardrobe uw LEFT JOIN colors c ON uw.color_id = c.color_id WHERE uw.item_id = %s", [top_item_id])
            row = cur.fetchone()
            if row:
                top_path = os.path.join(app.root_path, row['image_url'])
                desc = f"{row['color_name'] or ''} {row['item_name']}"
                
                final_url = call_fal_api(current_path, top_path, "atas", desc)
                
                temp_step = os.path.join(TEMP_FOLDER, f"step1_{user_id}.jpg")
                if download_image_from_url(final_url, temp_step):
                    current_path = temp_step

        if bottom_item_id and bottom_item_id != 'null':
            cur.execute("SELECT image_url, item_name, c.color_name FROM user_wardrobe uw LEFT JOIN colors c ON uw.color_id = c.color_id WHERE uw.item_id = %s", [bottom_item_id])
            row = cur.fetchone()
            if row:
                bottom_path = os.path.join(app.root_path, row['image_url'])
                desc = f"{row['color_name'] or ''} {row['item_name']}"
                
                final_url = call_fal_api(current_path, bottom_path, "bawah", desc)

        if final_url:
            return jsonify({
                "status": "success", 
                "result_url": final_url
            })
        else:
            return jsonify({"status": "error", "message": "Gagal memproses gambar"}), 400

    except Exception as e:
        print(f"❌ Error Process Outfit: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cur.close()

@app.route('/api/save-outfit', methods=['POST'])
def save_outfit_final():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        outfit_name = data.get('outfit_name')
        result_url = data.get('result_url')
        item_ids = data.get('item_ids')

        filename = f"outfit_{uuid.uuid4().hex}.jpg"
        final_path = os.path.join(UPLOAD_FOLDER, filename)
        
        response = requests.get(result_url)
        with open(final_path, 'wb') as f:
            f.write(response.content)
        
        db_path = f"static/uploads/{filename}"

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO outfits (user_id, outfit_name, result_image_url) VALUES (%s, %s, %s)", 
                    (user_id, outfit_name, db_path))
        outfit_id = cur.lastrowid

        for i_id in item_ids:
            if i_id:
                cur.execute("INSERT INTO outfit_items (outfit_id, item_id) VALUES (%s, %s)", (outfit_id, i_id))

        mysql.connection.commit()
        cur.close()
        return jsonify({"status": "success", "message": "Outfit berhasil disimpan!"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route('/api/wardrobe/simple/<int:user_id>', methods=['GET'])
def get_wardrobe_simple(user_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT 
                uw.item_id, 
                uw.image_url, 
                uw.style, 
                COALESCE(c_parent.category_name, c_child.category_name) as category_name 
            FROM user_wardrobe uw
            JOIN categories c_child ON uw.category_id = c_child.category_id
            LEFT JOIN categories c_parent ON c_child.parent_category_id = c_parent.category_id
            WHERE uw.user_id = %s
        """, [user_id])
        
        items = cur.fetchall()
        cur.close()
        return jsonify(success=True, items=items)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/my-outfits/<int:user_id>', methods=['GET'])
def get_my_outfits(user_id):
    try:
        cur = mysql.connection.cursor()
        query = """
            SELECT outfit_id, outfit_name, result_image_url, created_at 
            FROM outfits 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """
        cur.execute(query, [user_id])
        rows = cur.fetchall()
        cur.close()
        return jsonify(success=True, outfits=rows)
    except Exception as e:
        print(f"❌ Error My Outfit: {str(e)}")
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/share', methods=['POST'])
def share_outfit():
    data = request.get_json()
    user_id = data.get('user_id')
    image_url = data.get('image_url')
    caption = data.get('caption', '')

    if not user_id or not image_url:
        return jsonify({"message": "Invalid data"}), 400

    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO share_outfit (user_id, image_url, caption) VALUES (%s, %s, %s)",
        (user_id, image_url, caption)
    )
    mysql.connection.commit()
    cur.close()

    return jsonify({"message": "Shared successfully", "status": "success"})

@app.route('/api/explore/<int:viewer_id>', methods=['GET'])
def explore(viewer_id):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT 
            so.share_id,
            u.username,
            u.profile_photo_url,
            so.image_url,
            so.caption,
            so.created_at,
            (SELECT COUNT(*) FROM likes WHERE share_id = so.share_id) as total_likes,
            IF(l.like_id IS NULL, FALSE, TRUE) AS liked
        FROM share_outfit so
        JOIN users u ON so.user_id = u.user_id
        LEFT JOIN likes l ON l.share_id = so.share_id AND l.user_id = %s
        ORDER BY so.created_at DESC
    """, (viewer_id,))
    
    rows = cur.fetchall()
    cur.close()
    return jsonify(rows)

@app.route('/api/share/like', methods=['POST'])
def toggle_like():
    data = request.get_json()
    user_id = data['user_id']
    share_id = data['share_id']

    cur = mysql.connection.cursor()
    cur.execute("SELECT like_id FROM likes WHERE user_id=%s AND share_id=%s", (user_id, share_id))
    liked = cur.fetchone()

    if liked:
        cur.execute("DELETE FROM likes WHERE user_id=%s AND share_id=%s", (user_id, share_id))
        mysql.connection.commit()
        status = False
    else:
        cur.execute("INSERT INTO likes (user_id, share_id) VALUES (%s, %s)", (user_id, share_id))
        mysql.connection.commit()
        status = True
    
    cur.close()
    return jsonify({"liked": status})

@app.route('/api/wardrobe_action', methods=['GET', 'POST'])
def wardrobe_action():
    action = None
    user_id = None
    data = {}

    if request.method == 'GET':
        action = request.args.get('action')
        user_id = request.args.get('user_id')
    else:
        data = request.get_json() if request.is_json else request.form
        
        action = request.args.get('action') or data.get('action')
        user_id = data.get('user_id')

    cur = mysql.connection.cursor()

    try:
        if action == 'search':
            keyword = request.args.get('keyword', '')
            search_term = f"%{keyword}%"

            cur.execute("""
                SELECT item_name, image_url, style 
                FROM user_wardrobe 
                WHERE user_id=%s AND type='owned' AND item_name LIKE %s
            """, (user_id, search_term))
            wardrobe_res = cur.fetchall()

            cur.execute("""
                SELECT 
                    oc.item_name, 
                    oc.image_url, 
                    oc.purchase_link, 
                    COALESCE(oc.price, 0) as price,
                    IF(uw.item_id IS NOT NULL, 1, 0) as is_liked
                FROM online_catalog oc
                LEFT JOIN user_wardrobe uw 
                    ON oc.item_name = uw.item_name 
                    AND uw.user_id = %s 
                    AND uw.type = 'wishlist'
                WHERE oc.item_name LIKE %s
                LIMIT 20
            """, (user_id, search_term))
            
            online_rows = cur.fetchall()
            
            online_res = []
            for row in online_rows:
                online_res.append({
                    "item_name": row['item_name'],
                    "image_url": row['image_url'],
                    "purchase_link": row['purchase_link'],
                    "price": int(row['price']),
                    "is_liked": bool(row['is_liked'])
                })

            return jsonify({
                "status": True,
                "wardrobe": wardrobe_res,
                "online": online_res
            })

        elif action == 'toggle_wishlist':
            item_name = data.get('item_name')
            
            if not item_name or not user_id:
                return jsonify({"status": False, "message": "Data tidak lengkap"}), 400

            image_url = data.get('image_url', '')
            link = data.get('link', '')
            price = data.get('price', 0)

            cur.execute("""
                SELECT item_id FROM user_wardrobe 
                WHERE user_id=%s AND item_name=%s AND type='wishlist'
            """, (user_id, item_name))
            existing = cur.fetchone()

            if existing:
                cur.execute("DELETE FROM user_wardrobe WHERE item_id=%s", [existing['item_id']])
                msg = "Dihapus dari Favorit"
                is_liked = False
            else:
                cur.execute("""
                    INSERT INTO user_wardrobe 
                    (user_id, item_name, image_url, price, link, type, style, added_at)
                    VALUES (%s, %s, %s, %s, %s, 'wishlist', 'Casual', NOW())
                """, (user_id, item_name, image_url, price, link))
                msg = "Disimpan ke Favorit"
                is_liked = True
            
            mysql.connection.commit()
            return jsonify({"status": True, "message": msg, "is_liked": is_liked})

        elif action == 'get_favorites':
            cur.execute("""
                SELECT item_id, item_name, image_url, price, link 
                FROM user_wardrobe 
                WHERE user_id=%s AND type='wishlist' 
                ORDER BY added_at DESC
            """, [user_id])
            favorites = cur.fetchall()
            return jsonify({"status": True, "data": favorites})

        else:
            return jsonify({"status": False, "message": f"Action '{action}' tidak valid"}), 400

    except Exception as e:
        print(f"Error API: {e}")
        return jsonify({"status": False, "message": str(e)}), 500
    finally:
        cur.close()

@app.route('/api/user/<int:user_id>/posts', methods=['GET'])
def get_user_posts(user_id):
    try:
        cur = mysql.connection.cursor()
        
        query = """
            SELECT 
                so.share_id, 
                so.image_url, 
                so.caption, 
                so.created_at,
                (SELECT COUNT(*) FROM likes WHERE share_id = so.share_id) as total_likes
            FROM share_outfit so
            WHERE so.user_id = %s 
            ORDER BY so.share_id DESC
        """
        
        cur.execute(query, [user_id])
        posts = cur.fetchall()
        cur.close()
        
        return jsonify({"success": True, "posts": posts}), 200
    except Exception as e:
        print(f"❌ Error Get Posts: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500
    
@app.route('/api/delete/post', methods=['POST'])
def delete_post():
    try:
        data = request.get_json()
        share_id = data.get('id')
        
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM likes WHERE share_id=%s", [share_id])
        cur.execute("DELETE FROM share_outfit WHERE share_id=%s", [share_id])
        
        mysql.connection.commit()
        cur.close()
        return jsonify(success=True, message="Postingan berhasil dihapus")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/delete/outfit', methods=['POST'])
def delete_outfit():
    try:
        data = request.get_json()
        outfit_id = data.get('id')
        
        cur = mysql.connection.cursor()
       
        cur.execute("DELETE FROM outfit_items WHERE outfit_id=%s", [outfit_id])
        cur.execute("DELETE FROM outfits WHERE outfit_id=%s", [outfit_id])
        
        mysql.connection.commit()
        cur.close()
        return jsonify(success=True, message="Outfit berhasil dihapus")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/delete/wardrobe', methods=['POST'])
def delete_wardrobe_item():
    try:
        data = request.get_json()
        item_id = data.get('id') 
        
        cur = mysql.connection.cursor()
       
        cur.execute("DELETE FROM user_wardrobe WHERE item_id=%s", [item_id])
        mysql.connection.commit()
        cur.close()
        return jsonify(success=True, message="Item berhasil dihapus")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500
       
@app.route('/api/change_password', methods=['POST'])
def change_password():
    data = request.get_json()
    user_id = data.get('user_id')
    new_password = data.get('new_password')

    if not user_id or not new_password:
        return jsonify(success=False, message="Data tidak lengkap"), 400

    # Tambahkan Validasi Minimal & Maksimal
    if len(new_password) < 8 or len(new_password) > 12:
        return jsonify(success=False, message="Password harus antara 8 sampai 12 karakter"), 400
    
    cur = mysql.connection.cursor()
    
    cur.execute("SELECT last_password_change FROM users WHERE user_id=%s", [user_id])
    row = cur.fetchone()
    
    if row and row['last_password_change']:
        last_change = row['last_password_change']
        # Hitung selisih hari
        delta = datetime.now() - last_change
        if delta.days < 3:
            sisa_jam = 72 - (delta.total_seconds() / 3600)
            return jsonify(success=False, message=f"Tunggu {int(sisa_jam)} jam lagi untuk ganti password"), 403

    hashed = generate_password_hash(new_password)
    cur.execute("""
        UPDATE users 
        SET password_hash=%s, last_password_change=NOW() 
        WHERE user_id=%s
    """, (hashed, user_id))
    
    mysql.connection.commit()
    cur.close()
    
    return jsonify(success=True, message="Password berhasil diubah")

@app.route('/api/link_google', methods=['POST'])
def link_google():
    data = request.get_json()
    user_id = data.get('user_id')
    google_email = data.get('google_email')

    if not user_id or not google_email:
        return jsonify(success=False, message="Email Google diperlukan"), 400

    cur = mysql.connection.cursor()

    cur.execute("SELECT user_id FROM users WHERE email=%s AND user_id != %s", (google_email, user_id))
    existing = cur.fetchone()
    if existing:
        cur.close()
        return jsonify(success=False, message="Email Google ini sudah terpakai di akun lain"), 409

    cur.execute("UPDATE users SET email=%s WHERE user_id=%s", (google_email, user_id))
    mysql.connection.commit()
    cur.close()

    return jsonify(success=True, message="Akun Google berhasil ditautkan")
    
@app.route('/api/user/<int:user_id>/frequent', methods=['GET'])
def get_frequent_items(user_id):
    try:
        cur = mysql.connection.cursor()
        query = """
            SELECT c.category_name, COUNT(*) as freq
            FROM outfit_items oi
            JOIN outfits o ON oi.outfit_id = o.outfit_id
            JOIN user_wardrobe uw ON oi.item_id = uw.item_id
            JOIN categories c ON uw.category_id = c.category_id
            WHERE o.user_id = %s
            GROUP BY c.category_name
            ORDER BY freq DESC
            LIMIT 5
        """
        cur.execute(query, [user_id])
        rows = cur.fetchall()
        cur.close()

        items = [row['category_name'].title() for row in rows]

        if not items:
            items = ["Casual", "Formal", "Sport"]

        return jsonify(success=True, items=items)
    except Exception as e:
        print(f"Error frequent: {e}")
        return jsonify(success=False, items=["Error"]), 500
    
@app.route('/api/feedback/submit', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    user_id = data.get('user_id')
    message = data.get('message')
    rating = data.get('rating')

    if not all([user_id, message, rating]):
        return jsonify(success=False, message="Data tidak lengkap"), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO feedback (user_id, message, rating) VALUES (%s, %s, %s)",
            (user_id, message, rating)
        )
        mysql.connection.commit()
        cur.close()
        return jsonify(success=True, message="Feedback berhasil disimpan")
    except Exception as e:
        print(f"Error Feedback: {e}")
        return jsonify(success=False, message=str(e)), 500
    
@app.route('/api/debug/db-structure', methods=['GET'])
def debug_db_structure():
    return jsonify(success=True, message="Endpoint aktif")

def get_db():
    # Karena kamu pakai flask_mysqldb, gunakan connection yang sudah ada
    return mysql.connection
# ==========================================
# 1. LOGIN & AUTH (TERMASUK GOOGLE)
# ==========================================
@app.route('/')
def index():
    if 'admin_logged_in' in session: return redirect(url_for('admin_dashboard'))
    return redirect(url_for('admin_login'))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        
        # Gunakan cursor dari mysql extension
        cur = mysql.connection.cursor() 
        cur.execute("SELECT * FROM admins WHERE username=%s AND password=%s", (user, pw))
        admin = cur.fetchone()
        cur.close()
        
        if admin:
            session['admin_logged_in'] = True
            session['username'] = admin['username']
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Username atau Password Salah!")
            return redirect(url_for('admin_login'))     
    return render_template('login.html')

# --- INI YANG MENANGKAP RESPONS DARI GOOGLE ---
@app.route('/login/google', methods=['POST'])
def login_google_verify():
    # Karena ini login Admin, kita set session langsung
    session['admin_logged_in'] = True
    session['username'] = "Google Admin"
    return jsonify({"success": True})

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))

# ==========================================
# 2. DASHBOARD & HALAMAN UTAMA
# ==========================================
@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_logged_in' not in session: 
        return redirect(url_for('admin_login'))
    
    # Gunakan helper yang sudah kita buat tadi
    cursor = get_db_cursor() 
    
    try:
        # 1. STATISTIK UTAMA
        cursor.execute("SELECT COUNT(*) as total FROM users")
        total_users = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM online_catalog")
        total_products = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM user_wardrobe")
        total_wardrobe = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM feedback")
        total_feedback = cursor.fetchone()['total']

        # 2. GRAFIK KATEGORI
        cursor.execute("SELECT category, COUNT(*) as jumlah FROM online_catalog GROUP BY category")
        cat_data = cursor.fetchall()
        chart_cat_labels = [row['category'] if row['category'] else 'Uncategorized' for row in cat_data]
        chart_cat_values = [row['jumlah'] for row in cat_data]

        # 3. GRAFIK AUTH
        cursor.execute("SELECT auth_provider, COUNT(*) as jumlah FROM users GROUP BY auth_provider")
        auth_data = cursor.fetchall()
        chart_auth_labels = [row['auth_provider'].upper() if row['auth_provider'] else 'LOCAL' for row in auth_data]
        chart_auth_values = [row['jumlah'] for row in auth_data]

        # 4. TABEL TERBARU
        cursor.execute("SELECT username, email, created_at, auth_provider FROM users ORDER BY created_at DESC LIMIT 5")
        recent_users = cursor.fetchall()
        
        cursor.execute("""
            SELECT f.message, f.rating, f.created_at, u.username 
            FROM feedback f 
            JOIN users u ON f.user_id = u.user_id 
            ORDER BY f.created_at DESC LIMIT 3
        """)
        recent_feedback = cursor.fetchall()

        # JANGAN ADA conn.close() di sini karena 'conn' tidak didefinisikan
        # Kursor akan tertutup otomatis saat request selesai
        
        return render_template('index.html', 
                               total_users=total_users,
                               total_products=total_products,
                               total_wardrobe=total_wardrobe,
                               total_feedback=total_feedback,
                               chart_cat_labels=chart_cat_labels,
                               chart_cat_values=chart_cat_values,
                               chart_auth_labels=chart_auth_labels,
                               chart_auth_values=chart_auth_values,
                               recent_users=recent_users,
                               recent_feedback=recent_feedback)

    except Exception as e:
        print(f"Error di Dashboard: {e}")
        return f"Terjadi kesalahan sistem: {e}", 500
@app.route('/admin/products')
def admin_products_page():
    if 'admin_logged_in' not in session: return redirect(url_for('admin_login'))
    return render_template('produk_online.html')

@app.route('/admin/users')
def admin_users():
    if 'admin_logged_in' not in session: 
        return redirect(url_for('admin_login'))
    
    # Ambil kursor pakai helper
    cursor = get_db_cursor()
    
    try:
        cursor.execute("SELECT user_id, username, email, gender, auth_provider, created_at FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()
        # JANGAN tulis cursor.close() atau conn.close() di sini.
        return render_template('manage_users.html', users=users)
    except Exception as e:
        print(f"Error: {e}")
        return "Gagal mengambil data user", 500
    
# ==========================================
# 3. FITUR WARDROBE (LEMARI USER)
# ==========================================
@app.route('/admin/wardrobe')
def admin_wardrobe():
    if 'admin_logged_in' not in session: return redirect(url_for('admin_login'))
    conn = get_db()
    cursor = conn.cursor()
    # Hitung item berdasarkan user_id (aman jika kolom id tidak ada)
    query = """
        SELECT u.user_id, u.username, u.email, u.gender, COUNT(w.user_id) as total_items 
        FROM users u 
        LEFT JOIN user_wardrobe w ON u.user_id = w.user_id 
        GROUP BY u.user_id 
        ORDER BY total_items DESC
    """
    cursor.execute(query)
    users = cursor.fetchall()
    return render_template('wardrobe_users.html', users=users)

@app.route('/admin/wardrobe/<int:user_id>')
def admin_wardrobe_detail(user_id):
    if 'admin_logged_in' not in session: return redirect(url_for('admin_login'))
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username, email FROM users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        return "User tidak ditemukan", 404
    
    cursor.execute("SELECT * FROM user_wardrobe WHERE user_id = %s ORDER BY added_at DESC", (user_id,))
    items = cursor.fetchall()
    return render_template('wardrobe_detail.html', items=items, user=user)

@app.route('/admin/wardrobe/delete/<int:item_id>', methods=['POST'])
def admin_delete_wardrobe_item(item_id):
    if 'admin_logged_in' not in session: return jsonify({"status": False})
    conn = get_db()
    cursor = conn.cursor()
    # Pastikan tabel user_wardrobe punya kolom 'id'
    try:
        cursor.execute("DELETE FROM user_wardrobe WHERE id = %s", (item_id,))
        conn.commit()
        return jsonify({"status": True, "message": "Item dihapus"})
    except Exception as e:
        return jsonify({"status": False, "message": str(e)})

# ==========================================
# 4. FITUR FEEDBACK (UMPAN BALIK)
# ==========================================
@app.route('/admin/feedback')
def admin_feedback():
    if 'admin_logged_in' not in session: return redirect(url_for('admin_login'))
    conn = get_db()
    cursor = conn.cursor()
    query = """
        SELECT f.*, u.username, u.email 
        FROM feedback f 
        JOIN users u ON f.user_id = u.user_id 
        ORDER BY f.created_at DESC
    """
    cursor.execute(query)
    feedbacks = cursor.fetchall()
    return render_template('feedback.html', feedbacks=feedbacks)

@app.route('/admin/feedback/delete/<int:id>', methods=['POST'])
def delete_feedback(id):
    if 'admin_logged_in' not in session: return jsonify({"status": False})
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM feedback WHERE id = %s", (id,))
    conn.commit()
    return jsonify({"status": True, "message": "Feedback dihapus"})

@app.route('/api/feedback', methods=['POST'])
def api_send_feedback():
    try:
        data = request.json
        user_id = data.get('user_id')
        message = data.get('message')
        rating = data.get('rating', 5)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (user_id, message, rating) VALUES (%s, %s, %s)", 
                       (user_id, message, rating))
        conn.commit()
        return jsonify({"status": True, "message": "Terima kasih!"})
    except Exception as e:
        return jsonify({"status": False, "message": str(e)}), 500

# ==========================================
# 1. SMART IMPORT CSV (VERSI FIX: PISAH KATEGORI & STYLE)
# ==========================================
@app.route('/admin/products/import_csv', methods=['POST'])
def import_csv():
    if 'admin_logged_in' not in session: return redirect(url_for('admin_login'))
    if 'file_csv' not in request.files: return redirect(url_for('admin_products_page'))
    file = request.files['file_csv']
    if file.filename == '': return redirect(url_for('admin_products_page'))

    try:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        
        # Ambil koneksi dan kursor
        conn = mysql.connection
        cursor = conn.cursor()
        
        berhasil = 0
        header_found = False
        col_map = {} 

        for row in csv_input:
            if not row: continue
            
            # --- 1. DETEKSI HEADER ---
            if not header_found:
                clean_row = [str(x).strip().lower() for x in row]
                if 'name' in clean_row and 'price' in clean_row:
                    header_found = True
                    col_map = {
                        'name': clean_row.index('name'),
                        'price': clean_row.index('price'),
                        'image': clean_row.index('image') if 'image' in clean_row else -1,
                        'link': clean_row.index('link') if 'link' in clean_row else -1,
                        'gender': clean_row.index('gender') if 'gender' in clean_row else -1,
                        'source': clean_row.index('source') if 'source' in clean_row else -1,
                        'color': clean_row.index('color') if 'color' in clean_row else -1,
                        'category': clean_row.index('category') if 'category' in clean_row else -1,
                        'category_text': clean_row.index('category_text') if 'category_text' in clean_row else -1
                    }
                continue 

            # --- 2. PROSES DATA BARIS ---
            try:
                item_name = row[col_map['name']].strip()
                raw_price = ''.join(filter(str.isdigit, row[col_map['price']].strip()))
                price = int(raw_price) if raw_price else 0
                
                # Sesuai screenshot kamu: brand, gender, image_url, purchase_link, category, category_text, color
                brand = row[col_map['source']].strip() if col_map['source'] != -1 else "Online"
                
                gender_raw = row[col_map['gender']].strip().lower() if col_map['gender'] != -1 else "unisex"
                gender = 'unisex'
                if any(x in gender_raw for x in ['women', 'female', 'wanita']): gender = 'female'
                elif any(x in gender_raw for x in ['men', 'male', 'pria']): gender = 'male'

                val_category = row[col_map['category']].strip() if col_map['category'] != -1 else "General"
                val_category_text = row[col_map['category_text']].strip() if col_map['category_text'] != -1 else "General"
                color_val = row[col_map['color']].strip() if col_map['color'] != -1 else ""
                p_link = row[col_map['link']].strip() if col_map['link'] != -1 else ""
                i_url = row[col_map['image']].strip() if col_map['image'] != -1 else ""

                # --- 3. INSERT KE DATABASE ---
                cursor.execute("""
                    INSERT INTO online_catalog 
                    (item_name, price, brand, gender, category, category_text, color, purchase_link, image_url) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (item_name, price, brand, gender, val_category, val_category_text, color_val, p_link, i_url))
                
                berhasil += 1
            except Exception as line_error:
                print(f"❌ Baris gagal ({item_name}): {line_error}")
                continue

        conn.commit()
        flash(f"Berhasil import {berhasil} produk!")
        
    except Exception as e:
        print(f"🔥 Error Import: {e}")
        flash(f"Error: {str(e)}")
    
    return redirect(url_for('admin_products_page'))

# ==========================================
# 2. API CRUD MANUAL (VERSI FIX: INPUT TERPISAH)
# ==========================================
@app.route('/api/products', methods=['GET', 'POST'])
def api_products_crud():
    conn = get_db()
    cursor = conn.cursor()

    if request.method == 'GET':
        cursor.execute("SELECT * FROM online_catalog ORDER BY catalog_item_id DESC")
        products = cursor.fetchall()
        return jsonify({"status": True, "data": products})

    if request.method == 'POST':
        try:
            action = request.form.get('action')
            
            if action in ['create', 'update']:
                name = request.form.get('item_name')
                price = request.form.get('price')
                brand = request.form.get('brand')
                gender = request.form.get('gender')
                link = request.form.get('purchase_link')
                color = request.form.get('color')
                
                # AMBIL DUA INPUT KATEGORI
                val_category = request.form.get('category')           # Jenis (Shirt)
                val_category_text = request.form.get('category_text') # Style (Casual)

                # Handle Gambar
                image_url = request.form.get('foto_lama', '')
                if 'gambar_file' in request.files:
                    file = request.files['gambar_file']
                    if file.filename != '':
                        filename = str(int(time.time())) + "_" + file.filename
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                        image_url = f"http://{request.host}/static/uploads/{filename}"

                if action == 'create':
                    cursor.execute("""
                        INSERT INTO online_catalog 
                        (item_name, price, brand, gender, category, category_text, color, purchase_link, image_url) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (name, price, brand, gender, val_category, val_category_text, color, link, image_url))
                
                else: # Update
                    cursor.execute("""
                        UPDATE online_catalog 
                        SET item_name=%s, price=%s, brand=%s, gender=%s, category=%s, category_text=%s, color=%s, purchase_link=%s, image_url=%s 
                        WHERE catalog_item_id=%s
                    """, (name, price, brand, gender, val_category, val_category_text, color, link, image_url, request.form.get('id')))
                
                conn.commit()
                return jsonify({"status": True, "message": "Berhasil disimpan"})

            if action == 'delete':
                id = request.form.get('id')
                cursor.execute("DELETE FROM online_catalog WHERE catalog_item_id=%s", (id,))
                conn.commit()
                return jsonify({"status": True, "message": "Produk dihapus"})
                
        except Exception as e:
            return jsonify({"status": False, "message": str(e)})

    return jsonify({"status": False, "message": "Invalid Request"})

@app.route('/api/products/bulk_delete', methods=['POST'])
def bulk_delete_products():
    if 'admin_logged_in' not in session: return jsonify({"status": False}), 401
    data = request.json
    ids = data.get('ids', [])
    if not ids: return jsonify({"status": False})
    conn = get_db()
    cursor = conn.cursor()
    format_strings = ','.join(['%s'] * len(ids))
    cursor.execute(f"DELETE FROM online_catalog WHERE catalog_item_id IN ({format_strings})", tuple(ids))
    conn.commit()
    return jsonify({"status": True, "message": "Terhapus"})

@app.route('/api/users/bulk_delete', methods=['POST'])
def bulk_delete_users():
    if 'admin_logged_in' not in session: return jsonify({"status": False}), 401
    data = request.json
    ids = data.get('ids', [])
    if not ids: return jsonify({"status": False})
    conn = get_db()
    cursor = conn.cursor()
    format_strings = ','.join(['%s'] * len(ids))
    cursor.execute(f"DELETE FROM user_wardrobe WHERE user_id IN ({format_strings})", tuple(ids))
    cursor.execute(f"DELETE FROM users WHERE user_id IN ({format_strings})", tuple(ids))
    conn.commit()
    return jsonify({"status": True, "message": "Terhapus"})

# ==========================================
# 6. API FLUTTER (MOBILE)
# ==========================================
@app.route('/api/login', methods=['POST'])
def api_login_flutter():
    data = request.json
    email = data.get('email')
    password = data.get('password_hash') 
    is_google = data.get('is_google', False) 
    username = data.get('username', 'User') 
    gender = data.get('gender', 'male')

    conn = get_db()
    cursor = conn.cursor()

    if is_google:
        # --- LOGIKA LOGIN GOOGLE ---
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if not user:
            # REGISTER BARU VIA GOOGLE
            # Password otomatis diisi 'GOOGLE_AUTH' sebagai penanda
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, gender, auth_provider) 
                VALUES (%s, %s, 'GOOGLE_AUTH', %s, 'google')
            """, (username, email, gender))
            conn.commit()
            user_id = cursor.lastrowid
            return jsonify({"status": "success", "user_id": user_id, "username": username, "message": "Register Google Berhasil"})
        else:
            # USER LAMA LOGIN GOOGLE
            # Update password jadi 'GOOGLE_AUTH' untuk memastikan konsistensi
            if user['password_hash'] != 'GOOGLE_AUTH' or user['auth_provider'] != 'google':
                cursor.execute("UPDATE users SET password_hash='GOOGLE_AUTH', auth_provider='google' WHERE user_id=%s", (user['user_id'],))
                conn.commit()
            return jsonify({"status": "success", "user_id": user['user_id'], "username": user['username']})

    else:
        # --- LOGIKA LOGIN BIASA (EMAIL & PASSWORD) ---
        cursor.execute("SELECT * FROM users WHERE email = %s AND password_hash = %s", (email, password))
        user = cursor.fetchone()
        
        if user:
            return jsonify({"status": "success", "user_id": user['user_id'], "username": user['username']})
        return jsonify({"status": "error", "message": "Email atau Password Salah"}), 401

@app.route('/api/get_recommendation', methods=['GET'])
def get_recommendation():
    user_id = request.args.get('user_id')
    category = request.args.get('category')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT gender FROM users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    
    filters = []
    if user and user['gender'] in ['male', 'pria']: filters.append("(gender = 'male' OR gender = 'unisex')")
    elif user and user['gender'] in ['female', 'perempuan']: filters.append("(gender = 'female' OR gender = 'unisex')")
    else: filters.append("gender = 'unisex'")

    if category and category != 'All': filters.append(f"category = '{category}'")
    
    query = "SELECT * FROM online_catalog WHERE " + " AND ".join(filters) + " ORDER BY RAND() LIMIT 20"
    cursor.execute(query)
    products = cursor.fetchall()
    return jsonify({"status": True, "data": products})

@app.route('/api/wardrobe_action', methods=['GET', 'POST'])
def api_wardrobe_action():
    action = request.args.get('action')
    conn = get_db()
    cursor = conn.cursor()
    if action == 'search':
        kw = request.args.get('keyword')
        uid = request.args.get('user_id')
        cursor.execute("SELECT * FROM user_wardrobe WHERE user_id=%s AND item_name LIKE %s", (uid, f"%{kw}%"))
        wardrobe = cursor.fetchall()
        cursor.execute("SELECT * FROM online_catalog WHERE item_name LIKE %s LIMIT 10", (f"%{kw}%",))
        online = cursor.fetchall()
        return jsonify({"status": True, "wardrobe": wardrobe, "online": online})
    
    if action == 'toggle_wishlist' and request.method == 'POST':
        u_id = request.form.get('user_id')
        name = request.form.get('item_name')
        cursor.execute("SELECT * FROM user_wardrobe WHERE user_id=%s AND item_name=%s AND type='wishlist'", (u_id, name))
        if cursor.fetchone():
            cursor.execute("DELETE FROM user_wardrobe WHERE user_id=%s AND item_name=%s", (u_id, name))
            msg = "Dihapus dari Favorit"
        else:
            cursor.execute("INSERT INTO user_wardrobe (user_id, item_name, image_url, price, link, type) VALUES (%s,%s,%s,%s,%s,'wishlist')",
                           (u_id, name, request.form.get('image_url'), request.form.get('price'), request.form.get('link')))
            msg = "Disimpan ke Favorit"
        conn.commit()
        return jsonify({"status": True, "message": msg})
    return jsonify({"status": False})

@app.route('/api/users/crud', methods=['POST'])
def api_users_crud():
    if 'admin_logged_in' not in session: 
        return jsonify({"status": False, "message": "Unauthorized"}), 401

    data = request.json
    action = data.get('action')
    conn = get_db()
    cursor = conn.cursor()

    try:
        if action == 'create':
            # Cek email duplikat
            cursor.execute("SELECT user_id FROM users WHERE email = %s", (data.get('email'),))
            if cursor.fetchone():
                return jsonify({"status": False, "message": "Email sudah terdaftar!"})
            
            password = data.get('password')
            auth_type = data.get('auth_provider')

            # --- LOGIKA BARU SESUAI PERMINTAAN ---
            # Jika Admin pilih Google, Password OTOMATIS jadi 'GOOGLE_AUTH'
            if auth_type == 'google':
                password = "GOOGLE_AUTH"
            
            # Jika Admin pilih Email Biasa, Password WAJIB DIISI
            if auth_type == 'local' and (not password or password.strip() == ""):
                 return jsonify({"status": False, "message": "Password wajib diisi untuk user Email!"})

            cursor.execute("""
                INSERT INTO users (username, email, password_hash, gender, auth_provider) 
                VALUES (%s, %s, %s, %s, %s)
            """, (data.get('username'), data.get('email'), password, data.get('gender'), auth_type))
            
        elif action == 'update':
            user_id = data.get('user_id')
            password = data.get('password')
            auth_type = data.get('auth_provider')

            # Jika tipe diubah jadi Google, paksa password jadi 'GOOGLE_AUTH'
            if auth_type == 'google':
                cursor.execute("""
                    UPDATE users SET username=%s, email=%s, password_hash='GOOGLE_AUTH', gender=%s, auth_provider='google' 
                    WHERE user_id=%s
                """, (data.get('username'), data.get('email'), data.get('gender'), user_id))
            
            # Jika tipe Local, cek apakah admin ganti password atau tidak
            else:
                if password and password.strip() != "":
                    cursor.execute("""
                        UPDATE users SET username=%s, email=%s, password_hash=%s, gender=%s, auth_provider='local' 
                        WHERE user_id=%s
                    """, (data.get('username'), data.get('email'), password, data.get('gender'), user_id))
                else:
                    # Kalau password kosong, update info lain saja (password lama tetap)
                    cursor.execute("""
                        UPDATE users SET username=%s, email=%s, gender=%s, auth_provider='local' 
                        WHERE user_id=%s
                    """, (data.get('username'), data.get('email'), data.get('gender'), user_id))
        
        conn.commit()
        return jsonify({"status": True, "message": "Data berhasil disimpan"})
    
    except Exception as e:
        return jsonify({"status": False, "message": str(e)})
    finally:
        conn.close()


@app.route('/admin/shared-outfits')
def admin_shared_outfits():
    if 'admin_logged_in' not in session: return redirect(url_for('admin_login'))
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Join tabel share_outfit dengan users untuk dapat nama pengupload
    query = """
        SELECT s.*, u.username, u.email, u.gender 
        FROM share_outfit s 
        JOIN users u ON s.user_id = u.user_id 
        ORDER BY s.created_at DESC
    """
    cursor.execute(query)
    shares = cursor.fetchall()

    return render_template('shared_outfits.html', shares=shares)

@app.route('/admin/shared-outfits/delete/<int:id>', methods=['POST'])
def delete_shared_outfit(id):
    if 'admin_logged_in' not in session: return jsonify({"status": False})
    
    conn = get_db()
    cursor = conn.cursor()
    
    # (Opsional) Hapus file fisik gambar jika perlu
    # cursor.execute("SELECT image_url FROM share_outfit WHERE share_id = %s", (id,))
    # ... logika hapus file ...

    cursor.execute("DELETE FROM share_outfit WHERE share_id = %s", (id,))
    conn.commit()
    
    return jsonify({"status": True, "message": "Postingan dihapus"})

if __name__ == "__main__":
    print("\n✅ Server Berjalan! Siap menerima koneksi.")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)