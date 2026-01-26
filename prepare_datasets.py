import os
import shutil
from tqdm import tqdm # Untuk progress bar (install: pip install tqdm)

# --- Konfigurasi ---
ROOT_DIR = 'dataset' # Folder dataset asli Anda
OUTPUT_MEN_DIR = 'dataset_kategori_pria'
OUTPUT_WOMEN_DIR = 'dataset_kategori_wanita'
STYLES = ['Casual', 'Formal', 'Sport']
GENDERS = ['men', 'women']

# --- Fungsi Bantuan untuk Menyalin File ---
def flatten_dataset(source_dir, gender, output_dir):
    """
    Menyalin file dari struktur bersarang (nested) ke 
    struktur datar (flattened) berdasarkan kategori.
    """
    print(f"\nProcessing: {gender} -> {output_dir}")
    
    # Membuat folder output utama jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    total_files_copied = 0
    
    for style in STYLES:
        # Path ke folder gender di dalam style, misal: 'dataset/Casual/men'
        gender_path = os.path.join(source_dir, style, gender)
        
        if not os.path.exists(gender_path):
            print(f"-> Peringatan: Path tidak ditemukan, dilewati: {gender_path}")
            continue
            
        print(f"-> Mencari di: {gender_path}")
        
        # os.walk() akan menelusuri semua sub-folder secara rekursif
        # root: path saat ini (misal: 'dataset/Casual/men/pants/long')
        # dirs: daftar folder di dalam root
        # files: daftar file di dalam root
        for root, dirs, files in os.walk(gender_path):
            if not files:
                continue # Lewati jika tidak ada file di folder ini
                
            # Kita perlu mencari tahu nama kategori utamanya (pants, shirt, dll)
            # Caranya: dapatkan path relatif dari 'gender_path'
            # misal: root = 'dataset/Casual/men/pants/long'
            # rel_path = 'pants/long'
            rel_path = os.path.relpath(root, gender_path)
            
            # Kategori adalah folder pertama di rel_path
            # misal: 'pants/long' -> kategori = 'pants'
            # misal: 'shirt' -> kategori = 'shirt'
            category = rel_path.split(os.sep)[0]
            
            # Beberapa folder mungkin aneh (misal: 'jacket/Outdoor'), kita bersihkan
            if 'Outdoor' in category or 'Indoor' in category or 'indoor' in category or 'outdoor' in category:
                # Ambil folder sebelumnya, misal: 'jacket' dari 'jacket/Outdoor'
                # Ini adalah kasus khusus untuk dataset Sport Anda
                category = os.path.basename(os.path.dirname(root))

            if category == '.':
                continue # Lewati folder root itu sendiri
                
            # Buat folder target, misal: 'dataset_kategori_pria/pants'
            target_category_dir = os.path.join(output_dir, category)
            os.makedirs(target_category_dir, exist_ok=True)
            
            # Salin semua file dari 'root' ke 'target_category_dir'
            for file_name in files:
                source_file = os.path.join(root, file_name)
                target_file = os.path.join(target_category_dir, file_name)
                
                # Hindari duplikat jika nama file sama (opsional, tapi aman)
                if not os.path.exists(target_file):
                    shutil.copy2(source_file, target_file)
                    total_files_copied += 1

    print(f"Selesai! Total {total_files_copied} file disalin ke {output_dir}")

# --- Jalankan Skrip ---
if __name__ == "__main__":
    # 1. Proses data Pria
    flatten_dataset(ROOT_DIR, 'men', OUTPUT_MEN_DIR)
    
    # 2. Proses data Wanita
    flatten_dataset(ROOT_DIR, 'women', OUTPUT_WOMEN_DIR)
    
    print("\nSemua dataset telah dibuat!")
    print(f"Cek folder: {OUTPUT_MEN_DIR}")
    print(f"Cek folder: {OUTPUT_WOMEN_DIR}")