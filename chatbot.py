from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS  
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ✅ Allow CORS only from your frontend domain
CORS(app, origins=["https://msmeosem.in"])

# Load intents from JSON
with open("intents.json", "r") as file:
    data = json.load(file)

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

# Vectorize the input patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Load and save chat history in session
def load_chat_history():
    return session.get('chat_history', [])

def save_chat_history(chat_history):
    session['chat_history'] = chat_history

# Web interface route
@app.route('/', methods=['GET', 'POST'])
def home():
    chat_history = load_chat_history()
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        response = get_bot_response(query)
        chat_history.append({'role': 'user', 'text': query})
        chat_history.append({'role': 'bot', 'text': response})
        save_chat_history(chat_history)
    return render_template('index.html', chat_history=chat_history)

# Chat API - POST
@app.route("/chat", methods=["POST"])
def api_chat_post():
    if not request.is_json:
        return jsonify({"response": random.choice(responses.get("fallback", ["Invalid format."]))}), 415
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a valid message."})
    response = get_bot_response(user_input)
    return jsonify({"response": response})

# Chat API - GET
@app.route("/chat", methods=["GET"])
def api_chat_get():
    user_input = request.args.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please provide a message in the query parameter."})
    response = get_bot_response(user_input)
    return jsonify({"response": response})

# Clear chat history
@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop('chat_history', None)
    return jsonify({"response": "Chat history cleared!"})

# Logic for bot response
def get_bot_response(user_input):
    if not corpus:
        return "Bot training data is missing or invalid."
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, X)
    best_match_index = sim_scores.argmax()
    if sim_scores[0, best_match_index] > 0.2:
        best_intent = tags[best_match_index]
        return random.choice(responses[best_intent])
    else:
        return "Sorry, I couldn't understand that. Can you please rephrase?"

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
