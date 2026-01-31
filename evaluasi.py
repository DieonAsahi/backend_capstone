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
# 2. FUNGSI MMR (Maximal Marginal Relevance)
# ==========================================
def mmr_search(query_vec, doc_embeddings, top_k=1, diversity=0.5):
    """
    Algoritma MMR untuk memilih dokumen yang relevan tapi juga diverse.
    diversity: 0.0 (Murni Relevansi) - 1.0 (Murni Diversitas)
    """
    # 1. Hitung kesamaan query dengan semua dokumen
    word_doc_sim = cosine_similarity(query_vec, doc_embeddings).flatten()
    
    # 2. Filter kandidat awal (ambil top N by similarity dulu biar cepat)
    # Kita ambil 20 kandidat teratas berdasarkan similarity murni
    candidate_indices = np.argsort(word_doc_sim)[::-1][:20].tolist()
    
    # List untuk menyimpan index dokumen yang terpilih
    keywords_idx = []
    
    for _ in range(min(top_k, len(candidate_indices))):
        candidate_similarities = word_doc_sim[candidate_indices]
        
        # Jika belum ada yang dipilih, ambil yang similarity-nya paling tinggi
        if not keywords_idx:
            best_candidate = candidate_indices[np.argmax(candidate_similarities)]
        else:
            # Hitung kesamaan antara kandidat dengan dokumen yang SUDAH dipilih
            selected_embeddings = doc_embeddings[keywords_idx]
            candidate_embeddings = doc_embeddings[candidate_indices]
            
            # Sim matrix: (jumlah_kandidat x jumlah_terpilih)
            target_sim = cosine_similarity(candidate_embeddings, selected_embeddings)
            
            # Cari nilai max similarity kandidat terhadap dokumen yang sudah ada
            max_sim_to_selected = np.max(target_sim, axis=1)
            
            # RUMUS MMR:
            # MMR = (1 - diversity) * Sim(Query, Doc) - (diversity) * MaxSim(Doc, SelectedDocs)
            mmr_score = (1 - diversity) * candidate_similarities - (diversity) * max_sim_to_selected
            
            # Ambil index dengan skor MMR tertinggi
            best_candidate_local_idx = np.argmax(mmr_score)
            best_candidate = candidate_indices[best_candidate_local_idx]
            
        # Tambahkan ke daftar terpilih & hapus dari kandidat
        keywords_idx.append(best_candidate)
        candidate_indices.remove(best_candidate)
        
    return keywords_idx[0] # Kembalikan index dokumen terbaik (Top 1)

# ==========================================
# 3. FUNGSI TESTING
# ==========================================
def evaluasi_chatbot():
    print("\n🚀 MEMULAI EVALUASI PERFORMA (METODE MMR)...")
    print("------------------------------------------------")
    
    benar_retrieval = 0
    total_bleu_score = 0
    jumlah_data = len(df)
    chencherry = SmoothingFunction()

    # Kita tes ulang semua pertanyaan yang ada di dataset
    for index, row in df.iterrows():
        pertanyaan_asli = row['pertanyaan']
        jawaban_asli = row['jawaban']
        
        # 1. Encode Query
        query_vec = model.encode([pertanyaan_asli])
        
        # 2. Simulasikan Pencarian menggunakan MMR
        # Kita cek: Kalau ditanya A, apakah dia mengambil jawaban A menggunakan logika MMR?
        # diversity=0.5 adalah setting standar (balance)
        prediksi_idx = mmr_search(query_vec, embeddings, top_k=1, diversity=0.5)
        
        # 3. Hitung Akurasi Retrieval
        # Jika index prediksi == index asli, berarti dia mengambil data yang benar
        if prediksi_idx == index:
            benar_retrieval += 1
        
        # 4. Hitung BLEU Score (Kualitas Teks)
        jawaban_prediksi = df.iloc[prediksi_idx]['jawaban']
        
        reference = [jawaban_asli.lower().split()]
        candidate = jawaban_prediksi.lower().split()
        score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
        total_bleu_score += score

    # ==========================================
    # 4. HASIL AKHIR
    # ==========================================
    avg_accuracy = (benar_retrieval / jumlah_data) * 100
    avg_bleu = total_bleu_score / jumlah_data

    print(f"📊 Total Data Dites     : {jumlah_data}")
    print(f"⚙️ Metode Retrieval     : MMR (Maximal Marginal Relevance)")
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