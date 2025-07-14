# Movie Recommendation System

This is a content-based movie recommender system built using **machine learning**. It suggests similar movies based on their descriptions, genres, and numeric attributes like popularity and ratings.

## How It Works

We use a **hybrid content-based approach** combining:

- **TF-IDF** on movie overviews
- **One-hot encoding** on genre tags
- **Scaled numeric features**: popularity, vote average, and runtime

These features are merged and passed to a **K-Nearest Neighbors (KNN)** model using **cosine similarity** to recommend the most similar movies.

## ML Pipeline

- Genres are converted to one-hot using `MultiLabelBinarizer`
- Overviews are vectorized using `TfidfVectorizer`
- Numeric features are scaled using `MinMaxScaler`
- All features are combined using `scipy.hstack`
- `NearestNeighbors` from `sklearn` finds the closest movies

## Tech Stack

- Python, pandas, scikit-learn
- TF-IDF, KNN, cosine distance

## Website Link

- [Live Movie Recommendation Website](https://movie-rec-t4lj.onrender.com/)
