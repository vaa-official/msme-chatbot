import os
import logging
import random
import json
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Set secret key for session handling (your provided key)
app.secret_key = "a1129a14e9b670bc7e0eafc5b9586a1686abd3568fde30bb7b2138f70d67ae4a"

# Enable CORS for your website with session support
CORS(app, supports_credentials=True, origins=["https://msmeosem.in"])

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load intents data
try:
    with open("intents.json", "r") as file:
        data = json.load(file)
except Exception as e:
    logger.error(f"Error loading intents.json: {e}")
    data = []

# Prepare data for model
corpus = []
tags = []
responses = {}

for item in data:
    if "intent" in item and "patterns" in item and "responses" in item:
        intent = item["intent"]
        for pattern in item["patterns"]:
            corpus.append(pattern)
            tags.append(intent)
        responses[intent] = item["responses"]

# Initialize TfidfVectorizer if corpus is available
vectorizer = TfidfVectorizer()
if corpus:
    X = vectorizer.fit_transform(corpus)

# Session-based chat history functions
def load_chat_history():
    return session.get('chat_history', [])

def save_chat_history(chat_history):
    session['chat_history'] = chat_history

# Bot logic
def get_bot_response(user_input):
    if not corpus:
        return "Sorry, bot training data is missing."
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, X)
    best_match_index = sim_scores.argmax()
    if sim_scores[0, best_match_index] > 0.2:
        best_intent = tags[best_match_index]
        return random.choice(responses[best_intent])
    return random.choice(responses.get("fallback", ["Sorry, I couldn't understand that."]))

# Chat API - GET
@app.route("/chat", methods=["GET"])
def chat_get():
    user_input = request.args.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please provide a message."}), 400
    response = get_bot_response(user_input)
    chat_history = load_chat_history()
    chat_history.append({"role": "user", "text": user_input})
    chat_history.append({"role": "bot", "text": response})
    save_chat_history(chat_history)
    return jsonify({"response": response})

# Chat API - POST
@app.route("/chat", methods=["POST"])
def chat_post():
    if not request.is_json:
        return jsonify({"response": "Invalid request format. Must be JSON."}), 415
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a message."}), 400
    response = get_bot_response(user_input)
    chat_history = load_chat_history()
    chat_history.append({"role": "user", "text": user_input})
    chat_history.append({"role": "bot", "text": response})
    save_chat_history(chat_history)
    return jsonify({"response": response})

# Clear session history
@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop('chat_history', None)
    return jsonify({"response": "Chat history cleared."})

# Error handler
@app.errorhandler(500)
def handle_500(error):
    logger.error(f"Internal Server Error: {error}")
    return jsonify({"response": "An internal server error occurred."}), 500

# Run the app (only use this in development)
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
