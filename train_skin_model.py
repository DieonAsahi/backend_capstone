import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ===============================
# 1. Load Dataset
# ===============================
print("📂 Memuat dataset warna kulit...")
dataset_path = 'skin_dataset.csv'

if not os.path.exists(dataset_path):
    print(f"❌ Error: File {dataset_path} tidak ditemukan!")
    exit()

df = pd.read_csv(dataset_path)

# ===============================
# 2. Pisahkan Fitur & Label
# ===============================
X = df[['R', 'G', 'B']]
y = df['Label']

# ===============================
# 3. Normalisasi RGB (0–255 → 0–1)
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 4. Latih Model KNN
# ===============================
print("🧠 Melatih model KNN...")
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_scaled, y)

# ===============================
# 5. Simpan Model + Scaler
# ===============================
with open('skin_tone_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('skin_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model & scaler berhasil disimpan!")
print("🎨 Model siap digunakan di Flask.")