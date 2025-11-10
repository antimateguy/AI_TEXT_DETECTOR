from flask import Flask, request, jsonify, render_template
import joblib, re, string, numpy as np
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model + vectorizer and define features
xgb_model = joblib.load("./models/xgboost.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl") if \
              joblib.os.path.exists("./models/tfidf_vectorizer.pkl") else None
stop_words = set(stopwords.words("english"))

style_features = ['word_count','avg_word_len','punct_ratio','stopword_ratio','unique_ratio']

# Helper to extract numeric features
def extract_stylometric_features(text):
    words = re.findall(r'\b\w+\b', text.lower())
    num_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if num_words > 0 else 0
    punct_ratio = sum(ch in string.punctuation for ch in text) / (len(text)+1)
    stopword_ratio = len([w for w in words if w in stop_words]) / (num_words+1)
    unique_ratio = len(set(words)) / (num_words+1)
    return np.array([[num_words, avg_word_len, punct_ratio, stopword_ratio, unique_ratio]])

# Simple highlighting logic: mark top words by TF-IDF weight
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    X_text = vectorizer.transform([text]) if vectorizer else csr_matrix((1, 5000))
    X_num = extract_stylometric_features(text)
    X_comb = hstack([X_text, csr_matrix(X_num)])

    proba = xgb_model.predict_proba(X_comb)[0][1]
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
