import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import os
import shutil
import matplotlib.pyplot as plt

# --- KONFIGURASI FOLDER ---
# Script ini HANYA membaca folder dataset fitting yang baru
# Pastikan folder ini ada di proyek Anda
DATASET_ROOT = {
    'pria': 'dataset pria',     #
    'wanita': 'dataset wanita'
}

# --- DAFTAR TUGAS PELATIHAN ---
# Kita akan melatih 5 Model Fitting Baru
# Script akan otomatis menggabungkan data dari folder Casual/Formal/Sport
TRAINING_TASKS = [
    # 1. Pria - Baju (T-shirt/Shirt dijadikan satu logika Fit/Loose)
    {
        'name': 'model_fit_pria_baju',
        'gender': 'pria',
        'item_folder': 'Baju', # Sesuai nama folder Anda
        'styles': ['Casual', 'Formal', 'Sport'] 
    },
    # 2. Pria - Celana
    {
        'name': 'model_fit_pria_celana',
        'gender': 'pria',
        'item_folder': 'Celana', 
        'styles': ['Casual', 'Formal', 'Sport']
    },
    # 3. Wanita - Baju
    {
        'name': 'model_fit_wanita_baju',
        'gender': 'wanita',
        'item_folder': 'Baju',
        'styles': ['Casual', 'Formal', 'Sport']
    },
    # 4. Wanita - Celana
    {
        'name': 'model_fit_wanita_celana',
        'gender': 'wanita',
        'item_folder': 'Celana',
        'styles': ['Casual', 'Formal', 'Sport']
    },
    # 5. Wanita - Rok
    {
        'name': 'model_fit_wanita_rok',
        'gender': 'wanita',
        'item_folder': 'Rok',
        'styles': ['Casual', 'Formal', 'Sport']
    }
]

TEMP_DIR = 'temp_training_data'
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20

def prepare_data(task):
    """
    Fungsi PINTAR: Masuk ke folder Casual/Baju, Formal/Baju, Sport/Baju
    lalu mengambil semua gambar Fit/Loose untuk digabung jadi satu dataset latihan.
    """
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print(f"\n🔄 Mengumpulkan data untuk: {task['name']}...")
    
    total_images = 0
    base_path = DATASET_ROOT[task['gender']] 

    for style in task['styles']:
        # Path: dataset pria/Casual/Baju
        src_path = os.path.join(base_path, style, task['item_folder'])
        
        if not os.path.exists(src_path):
            print(f"   ⚠️ Folder style '{style}' tidak ditemukan, lanjut...")
            continue

        # Masuk ke folder fitting (Fit, Loose, dll)
        for label in os.listdir(src_path):
            label_path = os.path.join(src_path, label)
            
            if os.path.isdir(label_path):
                # Buat folder target di temp
                target_label_dir = os.path.join(TEMP_DIR, label)
                os.makedirs(target_label_dir, exist_ok=True)
                
                # Copy gambar
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Rename biar unik (misal: Casual_img1.jpg)
                        shutil.copy(
                            os.path.join(label_path, img_file),
                            os.path.join(target_label_dir, f"{style}_{img_file}") 
                        )
                        total_images += 1
    
    print(f"✅ Data siap! Total {total_images} gambar untuk {task['item_folder']}.")
    return total_images

def train_task(task):
    # 1. Siapkan Data (Agregasi)
    count = prepare_data(task)
    if count == 0:
        print(f"❌ SKIP: Tidak ada data untuk {task['name']}")
        return

    # 2. Augmentasi
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # 3. Load Data
    train_generator = train_datagen.flow_from_directory(
        TEMP_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        TEMP_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    labels = (train_generator.class_indices)
    print(f"🏷️ Label Fitting: {labels}")

    # 4. Model CNN
    model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(labels), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # 5. Training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS
    )

    # 6. Simpan Model Baru
    model_name = task['name'] + ".h5"
    model.save(model_name)
    print(f"💾 BERHASIL: {model_name} disimpan.")
    
    # Bersihkan folder sementara
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

# --- EKSEKUSI ---
if __name__ == "__main__":
    for task in TRAINING_TASKS:
        train_task(task)
    print("\n🎉 SEMUA 5 MODEL FITTING SELESAI DILATIH!")