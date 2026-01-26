import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# ==============================================================================
# 1. KONFIGURASI FOLDER UTAMA
# ==============================================================================
DATASET_ROOT = {
    'pria': 'dataset pria',    
    'wanita': 'dataset wanita'
}

TEMP_DIR = 'temp_training_data'
IMG_WIDTH, IMG_HEIGHT = 224, 224 
BATCH_SIZE = 32

# ==============================================================================
# 2. DEFINISI TASK (SESUAI STRUKTUR FOLDER ANDA)
# ==============================================================================
# Kita hanya perlu menunjuk folder 'Induk' (Baju/Celana/Rok)
# Script akan otomatis mengambil semua label (Fit, Loose, dll) di dalamnya.

TRAINING_TASKS = [
    # --- WANITA ---
    {
        'name': 'model_fit_wanita_baju.h5',
        'gender': 'wanita',
        # Ambil folder 'Baju' dari setiap style, lalu gabung isinya
        'sources': [
            ('Casual', 'Baju'), 
            ('Formal', 'Baju'), 
            ('Sport', 'Baju')
        ]
    },
    {
        'name': 'model_fit_wanita_celana.h5',
        'gender': 'wanita',
        'sources': [
            ('Casual', 'Celana'), 
            ('Formal', 'Celana'), 
            ('Sport', 'Celana')
        ]
    },
    {
        'name': 'model_fit_wanita_rok.h5',
        'gender': 'wanita',
        'sources': [
            ('Casual', 'Rok'), 
            ('Formal', 'Rok'), 
            ('Sport', 'Rok')
        ]
    },

    # --- PRIA ---
    {
        'name': 'model_fit_pria_baju.h5',
        'gender': 'pria',
        'sources': [
            ('Casual', 'Baju'), 
            ('Formal', 'Baju'), 
            ('Sport', 'Baju')
        ]
    },
    {
        'name': 'model_fit_pria_celana.h5',
        'gender': 'pria',
        'sources': [
            ('Casual', 'Celana'), 
            ('Formal', 'Celana'), 
            ('Sport', 'Celana')
        ]
    }
]

# ==============================================================================
# 3. FUNGSI CRAWLER (PENGUMPUL DATA)
# ==============================================================================
def prepare_data(task):
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print(f"\n🔄 [DATA PREP] Mengumpulkan gambar untuk: {task['name']}...")
    total_images = 0
    base_path = DATASET_ROOT[task['gender']] 

    # Loop setiap sumber (Style -> FolderItem)
    # Contoh: ('Casual', 'Baju')
    for style, folder_item in task['sources']:
        
        # Path: dataset wanita/Casual/Baju
        # Menggunakan fleksibilitas nama folder (kapital/kecil)
        possible_paths = [
            os.path.join(base_path, style, folder_item),
            os.path.join(base_path, style.capitalize(), folder_item),
            os.path.join(base_path, style, folder_item.capitalize())
        ]
        
        target_path = None
        for p in possible_paths:
            if os.path.exists(p):
                target_path = p
                break
        
        if not target_path:
            # print(f"   ⚠️ Folder tidak ditemukan: {style}/{folder_item} (Skip)")
            continue

        # Masuk ke folder Label (Fit, Loose, dll)
        # Struktur: .../Baju/Fit/gambar.jpg
        for label in os.listdir(target_path):
            label_path = os.path.join(target_path, label)
            
            if os.path.isdir(label_path):
                # Siapkan folder label di Temp
                temp_label_dir = os.path.join(TEMP_DIR, label) 
                os.makedirs(temp_label_dir, exist_ok=True)
                
                # Copy gambar
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        # Rename agar tidak bentrok: Casual_Baju_img1.jpg
                        new_name = f"{style}_{folder_item}_{img_file}"
                        shutil.copy(
                            os.path.join(label_path, img_file), 
                            os.path.join(temp_label_dir, new_name)
                        )
                        total_images += 1
    
    print(f"✅ Data Siap! Total {total_images} gambar terkumpul.")
    return total_images

# ==============================================================================
# 4. FUNGSI TRAINING (FINE TUNING PRO)
# ==============================================================================
def train_task(task):
    count = prepare_data(task)
    if count == 0: 
        print(f"❌ SKIP: Data kosong untuk {task['name']}")
        return

    # --- SETUP GENERATOR ---
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        TEMP_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        TEMP_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )

    labels = (train_gen.class_indices)
    num_classes = len(labels)
    print(f"🏷️ Label Terdeteksi: {list(labels.keys())}")

    # CLASS WEIGHTS
    cls_train = train_gen.classes
    if len(np.unique(cls_train)) > 1:
        class_weights = compute_class_weight('balanced', classes=np.unique(cls_train), y=cls_train)
        class_weights_dict = dict(enumerate(class_weights))
    else:
        class_weights_dict = None
    
    print(f"⚖️ Class Weights: {class_weights_dict}")

    # --- MEMBANGUN MODEL ---
    print("\n🧠 Membangun MobileNetV2...")
    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x) 
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model_name = task['name']
    checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, verbose=1)
    csv_logger = CSVLogger(model_name.replace('.h5', '_log.csv'))

    # --- FASE 1: WARM UP (10 Epoch) ---
    print(f"\n🔥 [{model_name}] FASE 1: Warm Up...")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    hist1 = model.fit(
        train_gen, validation_data=val_gen, epochs=10, 
        class_weight=class_weights_dict, callbacks=[checkpoint, csv_logger]
    )

    # --- FASE 2: FINE TUNING (20 Epoch) ---
    print(f"\n🧊 [{model_name}] FASE 2: Fine Tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]: 
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

    hist2 = model.fit(
        train_gen, validation_data=val_gen, epochs=20, 
        class_weight=class_weights_dict, callbacks=[checkpoint, early_stop, reduce_lr, csv_logger]
    )

    # --- VISUALISASI ---
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(acc, label='Train'); plt.plot(val_acc, label='Val'); plt.title(f'Akurasi: {model_name}'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(loss, label='Train'); plt.plot(val_loss, label='Val'); plt.title(f'Loss: {model_name}'); plt.legend()
    plt.savefig(model_name.replace('.h5', '_graph.png')); plt.close()

    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    print(f"✅ Selesai: {model_name}\n" + "="*50)

if __name__ == "__main__":
    for task in TRAINING_TASKS:
        train_task(task)
    print("\n🎉 SEMUA MODEL FITTING SELESAI DILATIH!")