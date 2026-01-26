import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI ---
# Ganti dua baris ini sesuai model yang ingin dilatih
# Opsi 1: Model Style
TRAIN_DATA_DIR = 'dataset'
MODEL_SAVE_PATH = 'model_style.h5'

# Opsi 2: Model Kategori Pria
# TRAIN_DATA_DIR = 'dataset_kategori_pria'
# MODEL_SAVE_PATH = 'model_kategori_pria.h5'

# Opsi 3: Model Kategori Wanita
# TRAIN_DATA_DIR = 'dataset_kategori_wanita'
# MODEL_SAVE_PATH = 'model_kategori_wanita.h5'

# --- 2. PENGATURAN UMUM ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 100  # Lebih tinggi, tapi dengan EarlyStopping

# --- 3. DATA GENERATOR ---
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
print(f"🔍 Found {num_classes} classes:")
print(train_generator.class_indices)

# --- 4. MEMBANGUN MODEL ---
print("🧠 Building model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Bekukan sebagian besar layer
for layer in base_model.layers[:-30]:  # Fine-tune 30 layer terakhir
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. KOMPILE MODEL ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 6. CALLBACKS ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6),
    CSVLogger(MODEL_SAVE_PATH.replace('.h5', '_log.csv'))
]

# --- 7. TRAINING ---
print(f"🚀 Starting training for {MODEL_SAVE_PATH} ...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

# --- 8. EVALUASI ---
val_loss, val_acc = model.evaluate(validation_generator)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"✅ Final Validation Loss: {val_loss:.4f}")

# --- 9. VISUALISASI HASIL ---
plot_save_path = MODEL_SAVE_PATH.replace('.h5', '_results.png')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.title('Model Accuracy'); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Model Loss'); plt.grid(True)

plt.tight_layout()
plt.savefig(plot_save_path)
print(f"📊 Saved training plot to {plot_save_path}")

print("\n🎉 Training complete! Best model saved to:")
print(f"➡️ {MODEL_SAVE_PATH}")
