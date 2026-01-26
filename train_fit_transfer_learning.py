import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 # [KUNCI SUKSES]
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
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
# [PENTING] MobileNetV2 butuh minimal 224x224 agar optimal
IMG_WIDTH, IMG_HEIGHT = 224, 224 
BATCH_SIZE = 32
EPOCHS = 30 

def prepare_data(task):
    """Logika Crawler Folder (Sama seperti sebelumnya)"""
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

    # --- 3. DATA GENERATOR DENGAN PREPROCESSING MOBILENET ---
    # MobileNet punya cara khusus memproses gambar (bukan cuma rescale 1/255)
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input, # [PENTING]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        TEMP_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        TEMP_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
    )

    labels = (train_generator.class_indices)
    print(f"🏷️ Label: {labels}")
    num_classes = len(labels)

    # --- 4. MEMBANGUN MODEL TRANSFER LEARNING (MOBILENETV2) ---
    print("🧠 Memuat MobileNetV2 (Pre-trained ImageNet)...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )

    # Bekukan layer dasar (biar ilmu lamanya gak hilang)
    base_model.trainable = False 

    # Tambahkan layer baru di atasnya
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x) # Dropout biar gak overfitting
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Gunakan Learning Rate kecil karena kita pakai Transfer Learning
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # --- 5. CALLBACKS (BIAR LEBIH PINTAR) ---
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    
    # Kalau akurasi mentok, turunkan learning rate otomatis
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    # Training
    print(f"🚀 Mulai melatih {task['name']}...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr]
    )

    # Simpan
    model.save(task['name'] + ".h5")
    pd.DataFrame(history.history).to_csv(task['name'] + "_log.csv", index=False)
    
    # Plot Grafik
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"Akurasi: {task['name']}"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss: {task['name']}"); plt.legend()
    plt.savefig(task['name'] + "_results.png"); plt.close()
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    print(f"✅ Selesai: {task['name']}")

if __name__ == "__main__":
    for task in TRAINING_TASKS:
        train_task(task)
    print("\n🎉 SEMUA MODEL FITTING SELESAI (DENGAN MOBILENET)!")