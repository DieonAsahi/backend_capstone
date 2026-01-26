import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import os

# ==========================================
# KONFIGURASI
# ==========================================
CSV_FILE = "dataset.csv"
MODEL_FILE = "rag_data.pkl" 
MODEL_NAME = 'all-MiniLM-L6-v2'  # Model AI yang cerdas & ringan

def train_model():
    print("⏳ Sedang memuat dataset...")
    
    # Cek apakah file dataset ada
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: File '{CSV_FILE}' tidak ditemukan di folder ini.")
        return

    try:
        df = pd.read_csv(CSV_FILE, on_bad_lines="skip")
        # Pastikan semua data dianggap teks
        df["pertanyaan"] = df["pertanyaan"].astype(str)
        df["jawaban"] = df["jawaban"].astype(str)
    except Exception as e:
        print(f"❌ Error membaca CSV: {e}")
        return

    print(f"🧠 Sedang mendownload & memuat otak AI '{MODEL_NAME}'...")
    print("(Proses ini butuh internet & agak lama di awal)")
    
    # Load Model AI (Sentence Transformer)
    # Ini bedanya! Kita pakai model yang sudah 'sekolah' membaca bahasa manusia
    model = SentenceTransformer(MODEL_NAME)

    print("🔄 Sedang mempelajari makna pertanyaan (Embeddings)...")
    # Mengubah kalimat jadi vektor makna (bukan sekadar hitung kata)
    embeddings = model.encode(df["pertanyaan"].tolist(), show_progress_bar=True)

    # Simpan Otak Baru
    data_to_save = {
        "df": df,
        "embeddings": embeddings,
        "model_name": MODEL_NAME
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"✅ SUKSES! Model disimpan di '{MODEL_FILE}'")
    print("👉 Sekarang jalankan 'python app.py' lagi.")

if __name__ == "__main__":
    train_model()