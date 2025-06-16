import os
from flask import Flask, request, jsonify, render_template
from recommendor import HybridRecommender  # Assuming your class is in recommender.py

app = Flask(__name__)

# Load the saved model once at startup
recommender = HybridRecommender.load("saved_model/")

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
        return jsonify({"input": title, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use provided PORT or default to 5000
    app.run(host='0.0.0.0', port=port)
