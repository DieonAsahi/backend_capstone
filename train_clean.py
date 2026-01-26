import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ==============================================================================
# KONFIGURASI (PILIH SALAH SATU & COMMENT YANG LAIN)
# ==============================================================================

# --- 1. UNTUK MODEL STYLE (Casual, Formal, Sport) ---
TRAIN_DATA_DIR = 'dataset'
MODEL_SAVE_PATH = 'model_style.h5'

# --- 2. UNTUK MODEL PRIA (Kemeja, Kaos, dll) ---
# TRAIN_DATA_DIR = 'dataset_kategori_pria'
# MODEL_SAVE_PATH = 'model_kategori_pria.h5'

# --- 3. UNTUK MODEL WANITA (Dress, Blouse, dll) ---
# TRAIN_DATA_DIR = 'dataset_kategori_wanita' 
# MODEL_SAVE_PATH = 'model_kategori_wanita.h5'

# Catatan: Fitting tidak dimasukkan sesuai permintaan Anda.

# ==============================================================================

# Pengaturan Training
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 50  # Batas maksimal, akan berhenti otomatis jika akurasi mentok
LEARNING_RATE = 0.0001 

def train_model():
    # Cek folder dataset
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"❌ Error: Folder '{TRAIN_DATA_DIR}' tidak ditemukan!")
        return

    print(f"🚀 Memulai pelatihan untuk: {MODEL_SAVE_PATH}")
    print(f"📂 Dataset: {TRAIN_DATA_DIR}")
    print("📊 Pembagian Data: 80% Training | 20% Validasi")

    # --- 1. DATA GENERATOR (Augmentasi) ---
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalisasi
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # <--- INI ARTINYA 20% DATA UNTUK VALIDASI
    )

    # Generator untuk 80% Training
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Generator untuk 20% Validasi
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"🔍 Terdeteksi {num_classes} Kelas: {list(train_generator.class_indices.keys())}")

    # --- 2. MEMBANGUN MODEL (MobileNetV2) ---
    # Input Layer Eksplisit (Solusi agar app.py tidak error batch_shape)
    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False  # Bekukan layer bawaan

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x) 
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # --- 3. KOMPILE MODEL ---
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- 4. CALLBACKS (LOGIKA AKURASI TERTINGGI) ---
    callbacks = [
        # Stop jika akurasi validasi tidak naik dalam 10 epoch
        EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True, # Kembalikan ke performa terbaik
            verbose=1,
            mode='max'
        ),
        # Simpan file h5 hanya saat mencapai rekor akurasi baru
        ModelCheckpoint(
            MODEL_SAVE_PATH, 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1,
            mode='max'
        ),
        # Kurangi learning rate jika stuck, biar lebih teliti mencari minimum loss
        ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.5, 
            patience=3, 
            verbose=1, 
            min_lr=1e-7,
            mode='max'
        ),
        CSVLogger(MODEL_SAVE_PATH.replace('.h5', '_log.csv'))
    ]

    # --- 5. EKSEKUSI TRAINING ---
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks
    )

    # --- 6. SIMPAN FINAL ---
    # Simpan ulang untuk memastikan file final benar-benar tersulis
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model berhasil disimpan: {MODEL_SAVE_PATH}")

    # --- 7. VISUALISASI ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plot_path = MODEL_SAVE_PATH.replace('.h5', '_graph.png')
    plt.savefig(plot_path)
    print(f"📊 Grafik hasil training disimpan: {plot_path}")

if __name__ == "__main__":
    train_model()