import os
from flask import Flask, request, jsonify, render_template
from recommendor import HybridRecommender  # Assuming your class is in recommender.py

app = Flask(__name__)

# Load the saved model once at startup
recommender = HybridRecommender.load("saved_model3/")

@app.route('/')
def index():
    return render_template("index.html")  # Optional: Add a UI

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    title = data.get("title", "")

    if not title:
        return jsonify({"error": "No movie title provided"}), 400

    try:
        recommendations = recommender.recommend(title, n=5)
        if isinstance(recommendations, dict) and "error" in recommendations:
            return jsonify(recommendations), 404
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/popular')
def get_popular():
    return jsonify(recommender.get_popular_movies(n=5))

@app.route('/latest')
def get_latest():
    return jsonify(recommender.get_latest_movies(n=5))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use provided PORT or default to 5000
    app.run(host='0.0.0.0', port=port,debug=True)
