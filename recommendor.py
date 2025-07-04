import pandas as pd
import numpy as np
import ast
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack,csr_matrix
import math
import requests
from dotenv import load_dotenv
import requests
import time
from sklearn.decomposition import TruncatedSVD
from difflib import get_close_matches


def fetch_movie_details_by_id(tmdb_id, api_key, retries=3, delay=0.25):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": api_key}

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            poster_path = data.get("poster_path")
            return {
                "poster_path": f"https://image.tmdb.org/t/p/w185{poster_path}" if poster_path else None
            }

        except requests.RequestException as e:
            print(f"[ERROR] Attempt {attempt+1}: Could not fetch poster for TMDB ID {tmdb_id}: {e}")
            time.sleep(delay * (2 ** attempt))  # Exponential backoff: 0.25s, 0.5s, 1s

    return {"poster_path": None}


class HybridRecommender:
    def __init__(self, path, save_dir="saved_model3/"):
        self.save_dir = save_dir
        self.df = pd.read_csv(path)
        self._prepare_data()
        self._build_model()
        self.save()

    def _prepare_data(self):
        # -- Genres to list --
        self.df['genres'] = self.df['genres'].apply(ast.literal_eval)

        # -- TF-IDF on overview (with stopwords removal) --
        self.df['overview'] = self.df['overview'].fillna("")
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.overview_vec = self.tfidf.fit_transform(self.df['overview'])
        print(self.overview_vec.shape)
        # self.svd = TruncatedSVD(n_components=600)  # Reduce overview vector from 10k → 300
        # self.overview_vec = self.svd.fit_transform(self.overview_vec)
        # self.overview_vec = csr_matrix(self.overview_vec)  # for hstack

        # -- Genre one-hot encoding --
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.genre_vec = self.mlb.fit_transform(self.df['genres'])

        # -- Numeric features (cleaned, scaled) --
        numeric_cols = ['popularity', 'vote_average', 'runtime']
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.scaler = MinMaxScaler()
        self.numeric_vec = csr_matrix(self.scaler.fit_transform(self.df[numeric_cols]))

        # -- Combine all sparse features --
        self.final_features = hstack([self.overview_vec, self.genre_vec, self.numeric_vec]).tocsr()

    def _build_model(self):
        self.nn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn.fit(self.final_features)

    def recommend(self, title, n=5):
        title = title.strip().lower()
        matches = self.df[self.df['title'].str.lower() == title]

        if matches.empty:
            all_titles = self.df['title'].str.lower().tolist()
            close_matches = get_close_matches(title, all_titles, n=3, cutoff=0.6)
            if close_matches:
                return {
                    "error": f"Movie '{title}' not found. Did you mean: {close_matches}?"
                }
            else:
                return {
                    "error": f"Movie '{title}' not found in the database."
                }
        idx = matches.index[0]
        vec = self.final_features.getrow(idx)  # safer than indexing directly
        distances, indices = self.nn.kneighbors(vec, n_neighbors=n+1)

        recommended_rows = self.df.iloc[indices[0][0:]]

        results = []
        for _, row in recommended_rows.iterrows():
            tmdb_id = row.get("movie_id")

            # Fallback: Fetch poster from TMDB if missing
            tmdb_data = fetch_movie_details_by_id(tmdb_id, self.tmdb_api_key)
            poster_path = tmdb_data.get("poster_path") or "https://via.placeholder.com/500x750?text=No+Image"

            results.append({
                "title": row["title"],
                "genres": row["genres"].split(", ") if isinstance(row["genres"], str) else [],
                "overview": row["overview"],
                "poster_path": poster_path,
                "release_date": row["release_date"],
                "runtime": int(row["runtime"]) if pd.notnull(row["runtime"]) and not math.isnan(row["runtime"]) else None,
                "vote_average": float(row["vote_average"]) if pd.notnull(row["vote_average"]) and not math.isnan(row["vote_average"]) else None
            })
            time.sleep(0.25)

        return results

        #return self.df.iloc[indices[0][1:]]['title'].tolist()

    def get_popular_movies(self, n):
        popular = self.df.copy()
        popular = popular.sort_values(by='popularity', ascending=False).head(n)
        results = []
        for _, row in popular.iterrows():
            tmdb_id = row.get("movie_id")

            # Fallback: Fetch poster from TMDB if missing
            tmdb_data = fetch_movie_details_by_id(tmdb_id, self.tmdb_api_key)
            poster_path = tmdb_data.get("poster_path") or "https://via.placeholder.com/500x750?text=No+Image"
            results.append({
                "title": row["title"],
                "genres": row["genres"].split(", ") if isinstance(row["genres"], str) else [],
                "overview": row["overview"],
                "poster_path": poster_path,
                "release_date": row["release_date"],
                "runtime": int(row["runtime"]) if pd.notnull(row["runtime"]) and not math.isnan(row["runtime"]) else None,
                "vote_average": float(row["vote_average"]) if pd.notnull(row["vote_average"]) and not math.isnan(row["vote_average"]) else None
            })
            time.sleep(0.25)

        return results

    def get_latest_movies(self, n):
        latest = self.df.copy()
        latest = latest[latest['release_date'].notna()]
        latest = latest.sort_values(by='release_date', ascending=False).head(n)
        results = []
        for _, row in latest.iterrows():
            tmdb_id = row.get("movie_id")

            # Fallback: Fetch poster from TMDB if missing
            tmdb_data = fetch_movie_details_by_id(tmdb_id, self.tmdb_api_key)
            poster_path = tmdb_data.get("poster_path") or "https://via.placeholder.com/500x750?text=No+Image"
            results.append({
                "title": row["title"],
                "genres": row["genres"].split(", ") if isinstance(row["genres"], str) else [],
                "overview": row["overview"],
                "poster_path": poster_path,
                "release_date": row["release_date"],
                "runtime": int(row["runtime"]) if pd.notnull(row["runtime"]) and not math.isnan(row["runtime"]) else None,
                "vote_average": float(row["vote_average"]) if pd.notnull(row["vote_average"]) and not math.isnan(row["vote_average"]) else None
            })
            time.sleep(0.25)
        return results

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        joblib.dump(self.nn, os.path.join(self.save_dir, "nn_model.joblib"))
        joblib.dump(self.tfidf, os.path.join(self.save_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.scaler, os.path.join(self.save_dir, "scaler.joblib"))
        joblib.dump(self.mlb, os.path.join(self.save_dir, "mlb.joblib"))
        joblib.dump(self.final_features, os.path.join(self.save_dir, "features.joblib"))
        self.df.to_csv(os.path.join(self.save_dir, "movie_metadata.csv"), index=False)

    @staticmethod
    def load(path="saved_model3/"):
        model = HybridRecommender.__new__(HybridRecommender)
        model.save_dir = path
        model.nn = joblib.load(os.path.join(path, "nn_model.joblib"))
        model.tfidf = joblib.load(os.path.join(path, "tfidf_vectorizer.joblib"))
        model.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
        model.mlb = joblib.load(os.path.join(path, "mlb.joblib"))
        model.final_features = joblib.load(os.path.join(path, "features.joblib"))
        model.df = pd.read_csv(os.path.join(path, "movie_metadata.csv"))
        load_dotenv()
        model.tmdb_api_key = os.getenv("TMDB_API_KEY") 

        return model
