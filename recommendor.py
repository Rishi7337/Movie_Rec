import pandas as pd
import numpy as np
import ast
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

class HybridRecommender:
    def __init__(self, path, save_dir="saved_model/"):
        self.save_dir = save_dir
        self.df = pd.read_csv(path)
        self._prepare_data()
        self._build_model()
        self.save()

    def _prepare_data(self):
        # Convert genre strings back to list
        self.df['genres'] = self.df['genres'].apply(ast.literal_eval)

        # TF-IDF on overview
        self.df['overview'] = self.df['overview'].fillna("")
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.overview_vec = self.tfidf.fit_transform(self.df['overview'])

        # Genre one-hot encoding
        self.mlb = MultiLabelBinarizer()
        self.genre_vec = self.mlb.fit_transform(self.df['genres'])

        # Numeric features
        numeric_cols = ['popularity', 'vote_average', 'runtime']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)

        self.scaler = MinMaxScaler()
        self.numeric_vec = self.scaler.fit_transform(self.df[numeric_cols])

        # Combine all features
        self.final_features = hstack([self.overview_vec, self.genre_vec, self.numeric_vec])

    def _build_model(self):
        self.nn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn.fit(self.final_features)

    def recommend(self, title, n=5):
        title = title.lower()
        matches = self.df[self.df['title'].str.lower() == title]
        if matches.empty:
            return ["Movie not found."]
        idx = matches.index[0]
        vec = self.final_features.getrow(idx)  # safer than indexing directly
        distances, indices = self.nn.kneighbors(vec, n_neighbors=n+1)
        return self.df.iloc[indices[0][1:]]['title'].tolist()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        joblib.dump(self.nn, os.path.join(self.save_dir, "nn_model.joblib"))
        joblib.dump(self.tfidf, os.path.join(self.save_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.scaler, os.path.join(self.save_dir, "scaler.joblib"))
        joblib.dump(self.mlb, os.path.join(self.save_dir, "mlb.joblib"))
        joblib.dump(self.final_features, os.path.join(self.save_dir, "features.joblib"))
        self.df.to_csv(os.path.join(self.save_dir, "movie_metadata.csv"), index=False)

    @staticmethod
    def load(path="saved_model/"):
        model = HybridRecommender.__new__(HybridRecommender)
        model.save_dir = path
        model.nn = joblib.load(os.path.join(path, "nn_model.joblib"))
        model.tfidf = joblib.load(os.path.join(path, "tfidf_vectorizer.joblib"))
        model.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
        model.mlb = joblib.load(os.path.join(path, "mlb.joblib"))
        model.final_features = joblib.load(os.path.join(path, "features.joblib"))
        model.df = pd.read_csv(os.path.join(path, "movie_metadata.csv"))
        return model
