import pandas as pd
import pickle
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download tokenizer untuk BLEU score
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================================
# 1. LOAD MODEL (OTAK YANG SUDAH DILATIH)
# ==========================================
MODEL_FILE = "rag_data.pkl"

print("⏳ Memuat model untuk dievaluasi...")
try:
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)
        df = data["df"]  # Database pertanyaan & jawaban asli
        embeddings = data["embeddings"] # Vektor ingatan AI
        model_name = data["model_name"]

    # Load Sentence Transformer
    model = SentenceTransformer(model_name)
    print("✅ Model berhasil dimuat.")

except FileNotFoundError:
    print(f"❌ Error: File '{MODEL_FILE}' tidak ditemukan. Jalankan 'model.py' dulu!")
    exit()

# ==========================================
# 2. FUNGSI TESTING
# ==========================================
def evaluasi_chatbot():
    print("\n🚀 MEMULAI EVALUASI PERFORMA...")
    print("------------------------------------------------")
    
    benar_retrieval = 0
    total_bleu_score = 0
    jumlah_data = len(df)
    chencherry = SmoothingFunction()

    # Kita tes ulang semua pertanyaan yang ada di dataset
    for index, row in df.iterrows():
        pertanyaan_asli = row['pertanyaan']
        jawaban_asli = row['jawaban']
        
        # 1. Simulasikan Pencarian (Retrieval)
        # Kita cek: Kalau ditanya A, apakah dia mengambil jawaban A?
        query_vec = model.encode([pertanyaan_asli])
        similarity = cosine_similarity(query_vec, embeddings).flatten()
        
        # Ambil index dengan skor tertinggi
        prediksi_idx = similarity.argmax()
        
        # 2. Hitung Akurasi Retrieval
        # Jika index prediksi == index asli, berarti dia mengambil data yang benar
        if prediksi_idx == index:
            benar_retrieval += 1
        
        # 3. Hitung BLEU Score (Kualitas Teks)
        # Kita bandingkan jawaban yang diambil sistem dengan jawaban kunci
        jawaban_prediksi = df.iloc[prediksi_idx]['jawaban']
        
        reference = [jawaban_asli.lower().split()]
        candidate = jawaban_prediksi.lower().split()
        score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
        total_bleu_score += score

    # ==========================================
    # 3. HASIL AKHIR
    # ==========================================
    avg_accuracy = (benar_retrieval / jumlah_data) * 100
    avg_bleu = total_bleu_score / jumlah_data

    print(f"📊 Total Data Dites     : {jumlah_data}")
    print(f"🎯 Retrieval Accuracy : {avg_accuracy:.2f}%")
    print(f"📝 Rata-rata BLEU Score: {avg_bleu:.4f}")
    print("------------------------------------------------")

    # Kesimpulan
    if avg_accuracy >= 90:
        print("✅ KESIMPULAN: Performa Sangat Bagus (Hapalan Kuat)")
    elif avg_accuracy >= 70:
        print("⚠️ KESIMPULAN: Performa Cukup (Perlu tambah data)")
    else:
        print("❌ KESIMPULAN: Performa Buruk (Cek dataset/model)")

if __name__ == "__main__":
    evaluasi_chatbot()