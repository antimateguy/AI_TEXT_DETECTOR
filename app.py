from flask import Flask, request, jsonify, render_template
import os, joblib, re, string, numpy as np
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

app = Flask(__name__)

# --- ✅ Reliable absolute path for Render and local use ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# --- Load the model safely ---
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except FileNotFoundError:
    print("❌ Model file not found! Make sure 'xgboost.pkl' is inside the models/ folder.")
    model = None

# --- Load vectorizer safely ---
if os.path.exists(VECTORIZER_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ TF-IDF vectorizer loaded.")
else:
    print("⚠️ Vectorizer not found; using dummy sparse matrix.")
    vectorizer = None

# --- Load stopwords ---
stop_words = set(stopwords.words("english"))

# --- Stylometric feature names ---
style_features = ['word_count','avg_word_len','punct_ratio','stopword_ratio','unique_ratio']

# --- Extract numeric stylistic features ---
def extract_stylometric_features(text):
    words = re.findall(r'\b\w+\b', text.lower())
    num_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if num_words > 0 else 0
    punct_ratio = sum(ch in string.punctuation for ch in text) / (len(text)+1)
    stopword_ratio = len([w for w in words if w in stop_words]) / (num_words+1)
    unique_ratio = len(set(words)) / (num_words+1)
    return np.array([[num_words, avg_word_len, punct_ratio, stopword_ratio, unique_ratio]])

# --- Simple highlighting logic for frontend ---
def highlight_text(text, label):
    words = re.findall(r'\b\w+\b', text)
    highlights = []
    for w in words:
        if len(w) < 4: 
            highlights.append(w)
            continue
        color_class = "ai" if label == "AI" and len(w)%3==0 else \
                      "human" if label == "Human" and len(w)%4==0 else ""
        if color_class:
            highlights.append(f'<span class="highlight {color_class}">{w}</span>')
        else:
            highlights.append(w)
    return " ".join(highlights)

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    # Combine TF-IDF + stylometric features
    X_text = vectorizer.transform([text]) if vectorizer else csr_matrix((1, 5000))
    X_num = extract_stylometric_features(text)
    X_comb = hstack([X_text, csr_matrix(X_num)])

    # Predict
    proba = model.predict_proba(X_comb)[0][1]
    label = "AI" if proba >= 0.5 else "Human"
    conf = proba if label == "AI" else 1 - proba

    highlighted = highlight_text(text, label)
    return jsonify({
        "label": label,
        "confidence": round(float(conf), 3),
        "highlighted_text": highlighted
    })

if __name__ == "__main__":
    app.run(debug=True)
