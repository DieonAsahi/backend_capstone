import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd             # [PENTING] Untuk simpan CSV
import matplotlib.pyplot as plt # [PENTING] Untuk gambar Grafik
import os
import shutil

# --- 1. KONFIGURASI FOLDER ---
DATASET_ROOT = {
    'pria': 'dataset pria',    
    'wanita': 'dataset wanita'
}

# --- 2. KONFIGURASI TASK ---
TRAINING_TASKS = [
    {'name': 'model_fit_pria_baju', 'gender': 'pria', 'item_folder': 'Baju', 'styles': ['Casual', 'Formal', 'Sport']},
    {'name': 'model_fit_pria_celana', 'gender': 'pria', 'item_folder': 'Celana', 'styles': ['Casual', 'Formal', 'Sport']},
    {'name': 'model_fit_wanita_baju', 'gender': 'wanita', 'item_folder': 'Baju', 'styles': ['Casual', 'Formal', 'Sport']},
    {'name': 'model_fit_wanita_celana', 'gender': 'wanita', 'item_folder': 'Celana', 'styles': ['Casual', 'Formal', 'Sport']},
    {'name': 'model_fit_wanita_rok', 'gender': 'wanita', 'item_folder': 'Rok', 'styles': ['Casual', 'Formal', 'Sport']}
]

TEMP_DIR = 'temp_training_data'
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 50 # Batas maksimal (nanti berhenti sendiri kalau pintar)

def prepare_data(task):
    """Mengumpulkan data dari folder bertingkat (Nested)"""
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print(f"\n🔄 Mengumpulkan data untuk: {task['name']}...")
    total_images = 0
    base_path = DATASET_ROOT[task['gender']] 

    for style in task['styles']:
        src_path = os.path.join(base_path, style, task['item_folder'])
        if not os.path.exists(src_path): continue

        for label in os.listdir(src_path):
            label_path = os.path.join(src_path, label)
            if os.path.isdir(label_path):
                target_label_dir = os.path.join(TEMP_DIR, label)
                os.makedirs(target_label_dir, exist_ok=True)
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy(os.path.join(label_path, img_file), os.path.join(target_label_dir, f"{style}_{img_file}"))
                        total_images += 1
    
    print(f"✅ Data siap! Total {total_images} gambar.")
    return total_images

def train_task(task):
    count = prepare_data(task)
    if count == 0: return

    # Augmentasi
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest', validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        TEMP_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        TEMP_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
    )

    labels = (train_generator.class_indices)
    print(f"🏷️ Label: {labels}")

    # Model
    model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(32, (3,3), activation='relu'), MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'), MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'), MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'), Dropout(0.5),
        Dense(len(labels), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    # Training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # --- 1. SIMPAN MODEL ---
    model_name = task['name'] + ".h5"
    model.save(model_name)
    print(f"💾 Model disimpan: {model_name}")

    # --- 2. SIMPAN LOG CSV ---
    log_name = task['name'] + "_log.csv"
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(log_name, index=False)
    print(f"📄 Log CSV disimpan: {log_name}")

    # --- 3. SIMPAN GRAFIK PNG ---
    plot_name = task['name'] + "_results.png"
    plt.figure(figsize=(12, 4))
    
    # Plot Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"Akurasi: {task['name']}")
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss: {task['name']}")
    plt.legend()

    plt.savefig(plot_name)
    plt.close()
    print(f"📊 Grafik PNG disimpan: {plot_name}")
    
    # Bersih-bersih
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    for task in TRAINING_TASKS:
        train_task(task)
    print("\n🎉 SEMUA MODEL, LOG, & GRAFIK SELESAI DIBUAT!")