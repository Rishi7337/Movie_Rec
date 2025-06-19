from collaborative_recommender import CollaborativeRecommender
import pandas as pd

# Train and save the model
cf = CollaborativeRecommender("./data/ratings.csv")

# Load model later
cf = CollaborativeRecommender.load("saved_cf_model/")

# Get user-specific recommendations
ratings_df = pd.read_csv("./data/ratings.csv")
all_movies = ratings_df['movie_id'].unique()
rated_by_user = ratings_df[ratings_df['user_id'] == 42]['movie_id'].tolist()

recommendations = cf.recommend_for_user(42, all_movies, rated_by_user)
print(recommendations)
