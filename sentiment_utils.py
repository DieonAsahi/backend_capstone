import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from imblearn.over_sampling import SMOTE, RandomOverSampler 

class SentimentAnalyzer:
    def __init__(self, csv_path='Lemon8_clean.csv'):
        self.csv_path = csv_path
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.knn = KNeighborsClassifier(n_neighbors=5) 
        self.stemmer = StemmerFactory().create_stemmer()
        self.is_trained = False
        
        # 1. DOWNLOAD STOPWORDS (Agar "dan", "yang" hilang)
        try:
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('indonesian'))
        except LookupError:
            print("Mendownload database stopwords NLTK...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('indonesian'))

        # 2. KAMUS KATA KUNCI MUTLAK (Safety Net)
        # Kata-kata ini akan membypass AI dan langsung memvonis sentimen
        self.negative_keywords = [
            'kecewa', 'jelek', 'buruk', 'parah', 'lemot', 'lambat', 'lag', 
            'force close', 'error', 'bug', 'rusak', 'nyesel', 'sampah', 
            'gagal', 'lelet', 'hang', 'bapuk', 'aneh', 'susah'
        ]
        self.positive_keywords = [
            'bagus', 'keren', 'mantap', 'suka', 'puas', 'membantu', 
            'recomended', 'love', 'terbaik', 'top', 'hebat', 'nyaman'
        ]

    def preprocess(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        # Hapus URL & Mention
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|\#\w+', '', text)
        # Hapus Simbol & Angka
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Hapus Stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        
        text = ' '.join(tokens)
        text = self.stemmer.stem(text)
        return text

    def train(self):
        if not os.path.exists(self.csv_path):
            print(f"Error: {self.csv_path} tidak ditemukan.")
            return False

        try:
            df = pd.read_csv(self.csv_path)
            df = df.dropna(subset=['text_clean', 'sentiment'])
            
            X_text = df['text_clean'].astype(str) 
            y = df['sentiment']
            
            # Cek Distribusi
            count = Counter(y)
            min_samples = min(count.values())

            # TF-IDF
            X_tfidf = self.vectorizer.fit_transform(X_text)

            # Balancing Data
            if min_samples < 2:
                sampler = RandomOverSampler(random_state=42)
            else:
                k_neighbors = min(5, min_samples - 1)
                sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)

            X_resampled, y_resampled = sampler.fit_resample(X_tfidf, y)
            self.knn.fit(X_resampled, y_resampled)
            
            self.is_trained = True
            print(f"Model Hybrid Siap! Dilatih dengan {len(y_resampled)} data.")
            return True

        except Exception as e:
            print(f"Training Error: {e}")
            return False

    def predict(self, texts):
        if not self.is_trained:
            self.train()

        results = []
        
        # Loop satu per satu untuk cek Kamus dulu
        for raw_text in texts:
            text_lower = raw_text.lower()
            prediction = None
            
            # CEK 1: Apakah ada kata negatif mutlak?
            for word in self.negative_keywords:
                if word in text_lower:
                    prediction = 'negatif'
                    break
            
            # CEK 2: Jika tidak ada negatif, cek positif mutlak?
            # (Opsional: Hati-hati, "tidak bagus" bisa terdeteksi bagus kalau logicnya salah. 
            # Tapi karena kita cek negatif dulu, "tidak bagus" aman karena biasanya tidak ada kata negatif kuat. 
            # Namun "kecewa" lebih prioritas).
            if prediction is None:
                # Kita biarkan AI menangani positif/netral agar lebih luwes, 
                # kecuali kita ingin memaksa kata tertentu.
                pass 

            # CEK 3: Jika Kamus tidak menemukan, BARU tanya ke AI (KNN)
            if prediction is None:
                clean_text = self.preprocess(raw_text)
                # Harus dalam bentuk list untuk transform
                vec = self.vectorizer.transform([clean_text]) 
                pred_knn = self.knn.predict(vec)[0]
                prediction = pred_knn
            
            results.append(prediction)
            
        return results