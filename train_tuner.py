# train_tuner.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras_tuner as kt # <-- Impor KerasTuner

# --- 1. KONFIGURASI ---
# Ganti dua baris ini sesuai model yang ingin dilatih
# Opsi 1: Model Style
# TRAIN_DATA_DIR = 'dataset'
# MODEL_SAVE_PATH = 'model_style.h5'

# Opsi 2: Model Kategori Pria
# TRAIN_DATA_DIR = 'dataset_kategori_pria'
# MODEL_SAVE_PATH = 'model_kategori_pria.h5'

# Opsi 3: Model Kategori Wanita
TRAIN_DATA_DIR = 'dataset_kategori_wanita'
MODEL_SAVE_PATH = 'model_kategori_wanita.h5'

# --- 2. PENGATURAN UMUM ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS_TUNING = 10    # Epochs untuk setiap percobaan (Grid Search)
EPOCHS_FINAL = 30     # Epochs untuk melatih model terbaik

# --- 3. DATA GENERATOR (Sama) ---
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
print(f"📂 Loading training data from: {TRAIN_DATA_DIR}")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
num_classes = len(train_generator.class_indices)
print(f"🔍 Found {num_classes} classes: {train_generator.class_indices}")

# --- 4. MEMBANGUN FUNGSI MODEL (PENTING UNTUK TUNER) ---
# Kita ubah model build kita menjadi sebuah fungsi
def build_model(hp):
    """Fungsi ini membangun model dan mendefinisikan hyperparameter."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    # Kita bekukan sebagian besar layer
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # --- Ini adalah "Grid Search" untuk Dropout ---
    # 'hp.Float' akan mencari nilai float terbaik antara 0.2 dan 0.5
    hp_dropout = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(hp_dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # --- Ini adalah "Grid Search" untuk Learning Rate ---
    # 'hp.Choice' akan memilih salah satu nilai dari daftar
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 5. INISIALISASI "GRID SEARCH" (Tuner) ---
print("⚙ Initializing KerasTuner (Grid Search)...")
# Kita gunakan RandomSearch, ini lebih cepat dari GridSearch (mencoba kombinasi acak)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy', # Tujuan: mencari val_accuracy tertinggi
    max_trials=5,             # Berapa banyak kombinasi yang ingin dicoba
    executions_per_trial=1,   # Berapa kali 1 kombinasi dijalankan
    directory='tuner_logs',   # Folder untuk menyimpan hasil
    project_name=f'tuning_{os.path.basename(TRAIN_DATA_DIR)}'
)

# Hapus log lama jika ada
tuner.results_summary()
print("🧹 Clearing old tuner logs...")

# --- 6. JALANKAN "GRID SEARCH" ---
print(f"🚀 Starting Hyperparameter Search (Tuning)...")
# Kita tambahkan EarlyStopping di sini agar pencarian lebih cepat
stop_early = EarlyStopping(monitor='val_loss', patience=3)

tuner.search(
    train_generator,
    epochs=EPOCHS_TUNING,
    validation_data=validation_generator,
    callbacks=[stop_early]
)

# --- 7. DAPATKAN MODEL TERBAIK ---
print("\n✅ Search complete. Getting best model...")
# Ambil hyperparameter (setelan) terbaik
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"🏆 Best Hyperparameters found:")
print(f"   - Dropout Rate: {best_hps.get('dropout_rate'):.2f}")
print(f"   - Learning Rate: {best_hps.get('learning_rate')}")

# Bangun model final dengan setelan terbaik
model = tuner.hypermodel.build(best_hps)
model.summary()

# --- 8. LATIH MODEL TERBAIK (FINAL) ---
print(f"\n🚀 Starting FINAL training with best model...")
history = model.fit(
    train_generator,
    epochs=EPOCHS_FINAL,
    validation_data=validation_generator,
    callbacks=[
        # Gunakan EarlyStopping lagi untuk pelatihan final
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        # Simpan model terbaik
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    ]
)

# --- 9. EVALUASI (Sama) ---
val_loss, val_acc = model.evaluate(validation_generator)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")

# --- 10. VISUALISASI HASIL (Sama) ---
plot_save_path = MODEL_SAVE_PATH.replace('.h5', '_results.png')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.legend(); plt.title('Model Accuracy'); plt.grid(True)
plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.legend(); plt.title('Model Loss'); plt.grid(True)
plt.tight_layout(); plt.savefig(plot_save_path);
print(f"📊 Saved training plot to {plot_save_path}")
print(f"\n🎉 Training complete! Best tuned model saved to: {MODEL_SAVE_PATH}")