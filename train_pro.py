import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# KONFIGURASI TASK (GANTI BAGIAN INI SESUAI YANG MAU DILATIH)
# ==============================================================================

# Contoh untuk Fitting (Karena tadi akurasinya rendah)
# TRAIN_DATA_DIR = 'dataset_fit_wanita_baju' # Ganti dengan folder dataset Anda
# MODEL_SAVE_PATH = 'model_fit_wanita_baju.h5'

# Contoh untuk Style (Bisa dipakai juga)
# TRAIN_DATA_DIR = 'dataset' 
# MODEL_SAVE_PATH = 'model_style.h5'

# Opsi 2: Model Kategori Pria
# TRAIN_DATA_DIR = 'dataset_kategori_pria'
# MODEL_SAVE_PATH = 'model_kategori_pria.h5'

# Opsi 3: Model Kategori Wanita
TRAIN_DATA_DIR = 'dataset_kategori_wanita'
MODEL_SAVE_PATH = 'model_kategori_wanita.h5'


# ==============================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def create_model(num_classes):
    # 1. Base Model (MobileNetV2)
    input_shape = IMG_SIZE + (3,)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Bekukan dulu

    # 2. Top Head (Otak Baru)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)  # Stabilizer
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)          # Mencegah Overfitting
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs), base_model

def train_pro():
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"❌ Folder dataset tidak ditemukan: {TRAIN_DATA_DIR}")
        return

    print(f"🚀 Memulai TRAINING PRO untuk: {MODEL_SAVE_PATH}")

    # --- 1. DATA GENERATOR (Augmentasi Halus) ---
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    print("📂 Memuat Data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    print(f"🔍 Kelas: {list(train_generator.class_indices.keys())}")

    # Hitung Class Weights (Penting jika data tidak seimbang)
    cls_train = train_generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(cls_train), y=cls_train)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"⚖️ Class Weights: {class_weights_dict}")

    # --- 2. BUILD MODEL ---
    model, base_model = create_model(num_classes)

    # Callback Umum
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    csv_logger = CSVLogger(MODEL_SAVE_PATH.replace('.h5', '_log.csv'))

    # ==========================================================================
    # PHASE 1: WARM UP (Melatih Layer Atas Saja)
    # ==========================================================================
    print("\n🔥 PHASE 1: WARM UP (Melatih Top Layer)...")
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    hist1 = model.fit(
        train_generator,
        epochs=10, # Cukup 10 epoch untuk pemanasan
        validation_data=val_generator,
        class_weight=class_weights_dict,
        callbacks=[checkpoint, csv_logger]
    )

    # ==========================================================================
    # PHASE 2: FINE TUNING (Melatih Detail Halus)
    # ==========================================================================
    print("\n🧊 PHASE 2: FINE TUNING (Mencairkan sebagian layer)...")
    
    # Cairkan 40 layer teratas MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    # Kompile ulang dengan Learning Rate SANGAT KECIL (agar ilmu lama tidak rusak)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks Tambahan untuk Fine Tuning
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

    hist2 = model.fit(
        train_generator,
        epochs=30, # Latih lebih lama
        validation_data=val_generator,
        class_weight=class_weights_dict,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger]
    )

    print(f"\n🎉 TRAINING SELESAI! Model terbaik disimpan di: {MODEL_SAVE_PATH}")

    # --- VISUALISASI GABUNGAN ---
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc'); plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=10, color='green', linestyle='--', label='Start Fine Tuning')
    plt.title('Accuracy History'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss'); plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=10, color='green', linestyle='--', label='Start Fine Tuning')
    plt.title('Loss History'); plt.legend()
    
    plt.savefig(MODEL_SAVE_PATH.replace('.h5', '_fine_tune_graph.png'))
    print("📊 Grafik training disimpan.")

if __name__ == "__main__":
    train_pro()