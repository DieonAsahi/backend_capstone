import pandas as pd
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. DEFINISI DATA (Sesuai database db_swipeer Anda)
# Skin Tone: 1:Putih, 2:Sawo Matang, 3:Gelap, 4:Kuning Langsat
skin_tone_ids = [1, 2, 3, 4]
# Body Shape: 1:Hourglass, 2:Pear, 3:Inverted Triangle, 4:Rectangle, 5:Apple
body_shape_ids = [1, 2, 3, 4, 5]
# Color IDs: 1 s/d 33 (berdasarkan tabel colors Anda)
color_ids = list(range(1, 34))
# Fit Type (Fitting_name): Diambil dari kategori anak di database Anda
fit_types = ['Fit', 'Loose', 'Fitted', 'Flare', 'Shoulder', 'Straight', 'Wide', 'Mermaid']
# Kategori Utama: tshirt, pants, shirt, outer, blazer, blouse, dress, skirt
category_ids = [1, 2, 3, 5, 6, 7, 8, 9]

def generate_smart_dataset(n=10000):
    data = []
    for _ in range(n):
        st = random.choice(skin_tone_ids)
        bs = random.choice(body_shape_ids)
        cl = random.choice(color_ids)
        ft = random.choice(fit_types)
        cat = random.choice(category_ids)
        
        score = 0
        # LOGIKA SKOR BERDASARKAN KNOWLEDGE BASE (recommendation_rules)
        # Warna vs Skin Tone
        if st == 1 and cl in [1, 2, 3, 6, 8, 11, 23]: score += 50  # Putih + Kontras
        if st == 4 and cl in [3, 7, 12, 14, 15, 17, 19]: score += 50 # Kuning Langsat + Earth Tone
        if st == 2 and cl in [6, 9, 11, 14, 16, 21, 25]: score += 50 # Sawo Matang + Cerah/Warm
        if st == 3 and cl in [6, 7, 9, 11, 16, 20, 21]: score += 50 # Gelap + Pop-up Colors
        
        # Fit vs Body Shape
        if bs == 2 and ft in ['Flare', 'Shoulder']: score += 40     # Pear
        if bs == 1 and ft in ['Fitted', 'Mermaid', 'Fit']: score += 40 # Hourglass
        if bs == 5 and ft in ['Loose', 'Straight']: score += 40      # Apple
        if bs == 3 and ft in ['Wide', 'Flare']: score += 40          # Inverted Triangle
        
        label = 1 if score >= 50 else 0
        data.append([st, bs, cl, ft, cat, label])
        
    return pd.DataFrame(data, columns=['skin_id', 'body_id', 'color_id', 'fit_type', 'cat_id', 'label'])

# --- EKSEKUSI PELATIHAN ---
print("Generating dataset...")
df = generate_smart_dataset(10000)

# Preprocessing Fit Type
le_fit = LabelEncoder()
df['fit_type'] = le_fit.fit_transform(df['fit_type'])

X = df.drop('label', axis=1)
y = df['label']

print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X, y)

# Simpan Model dan Encoder
joblib.dump(model, 'fashion_expert_model.pkl')
joblib.dump(le_fit, 'fit_encoder.pkl')
print("✅ Sukses! File 'fashion_expert_model.pkl' dan 'fit_encoder.pkl' telah dibuat.")