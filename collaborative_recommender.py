import pandas as pd
import os
import joblib
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic, dump

class KNNRecommender:
    def __init__(self, ratings_path, save_dir="saved_knn_model/", user_based=True):
        self.ratings_path = ratings_path
        self.save_dir = save_dir
        self.user_based = user_based
        self.model = None
        self.trainset = None
        self._prepare_data()
        self._train_model()
        self._save_model()

    def _prepare_data(self):
        df = pd.read_csv(self.ratings_path)
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
        self.trainset = data.build_full_trainset()

    def _train_model(self):
        sim_options = {
            'name': 'cosine',
            'user_based': self.user_based  # True for user-user, False for item-item
        }
        self.model = KNNBasic(sim_options=sim_options)
        self.model.fit(self.trainset)

    def _save_model(self):
        os.makedirs(self.save_dir, exist_ok=True)
        dump.dump(os.path.join(self.save_dir, "knn_model"), algo=self.model)
        joblib.dump(self.trainset, os.path.join(self.save_dir, "trainset.joblib"))
        print("Model and trainset saved to", self.save_dir)

    @staticmethod
    def load(path="saved_knn_model/"):
        recommender = KNNRecommender.__new__(KNNRecommender)
        recommender.save_dir = path
        _, recommender.model = dump.load(os.path.join(path, "knn_model"))
        recommender.trainset = joblib.load(os.path.join(path, "trainset.joblib"))
        return recommender
    
    def recommend_for_user(self, user_id, top_n):
    
        inner_id = self.model.trainset.to_inner_uid(user_id)
        neighbors = self.model.get_neighbors(inner_id, k=top_n)
        neighbors = (self.model.trainset.to_raw_uid(inner_id) for inner_id in neighbors)
        return list(neighbors)