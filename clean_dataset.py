#clean_dataset.py
import os
from PIL import Image
from tqdm import tqdm # Pastikan sudah install: pip install tqdm

# PENTING: Masukkan semua folder dataset Anda di sini
# clean_dataset.py (Bagian atas saja yang diubah)

# ... import library ...

DIRECTORIES_TO_CLEAN = [
    'dataset',                   # Model Style
    'dataset_kategori_pria',     # Model Kategori Pria
    'dataset_kategori_wanita',   # Model Kategori Wanita
    'dataset pria',              # [BARU] Dataset Fitting Pria
    'dataset wanita'             # [BARU] Dataset Fitting Wanita
]

# ... sisa kodenya sama persis ...

def clean_images(directory):
    print(f"\nMemindai folder: {directory}")
    corrupted_count = 0
    total_files = 0
    
    # Gunakan tqdm untuk progress bar
    pbar = tqdm(os.walk(directory), desc=f"Memindai {os.path.basename(directory)}")
    
    for root, dirs, files in pbar:
        for file_name in files:
            total_files += 1
            file_path = os.path.join(root, file_name)
            
            # Cek ekstensi file yang umum
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                print(f"-> Menghapus file non-gambar: {file_path}")
                os.remove(file_path)
                corrupted_count += 1
                continue

            # Coba buka file gambar untuk memverifikasi
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Memverifikasi integritas file
            except (IOError, Image.UnidentifiedImageError) as e:
                print(f"-> Menghapus file gambar rusak: {file_path} (Error: {e})")
                os.remove(file_path)
                corrupted_count += 1
                
    print(f"Selesai memindai {directory}. Total {corrupted_count} file rusak/ilegal dihapus dari {total_files} file.")

# --- Jalankan Pembersih ---
if __name__ == "__main__":
    for d in DIRECTORIES_TO_CLEAN:
        if os.path.exists(d):
            clean_images(d)
        else:
            print(f"Peringatan: Folder {d} tidak ditemukan, dilewati.")
    
    print("\n--- Pembersihan Selesai ---")
    print("Sekarang Anda bisa menjalankan 'train.py' lagi.")