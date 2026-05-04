import os
import joblib
import string
import re
import nltk
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Initializing Flask App ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for frontend communication

# --- Groq Client Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/favicon.png')
def favicon():
    return send_from_directory('.', 'favicon.png')

@app.route('/logo.png')
def logo():
    return send_from_directory('.', 'logo.png')

# --- NLTK Data Setup ---
# Download necessary NLTK data (only if not already present)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"NLTK Download Warning: {e}")

# --- Loading Model & Vectorizer ---
MODELS_DIR = os.path.join("mood_predict", "models")
MODEL_PATH = os.path.join(MODELS_DIR, 'production_model_v8.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'production_tfidf_v8.pkl')

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model assets: {e}")
    model = None
    tfidf = None

# --- Preprocessing Logic ---
lemmatizer = WordNetLemmatizer()

def pre_process(text):
    # Lowercase
    text = text.lower()
    # 2. Remove punctuation BEFORE handling negations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Handle negations by attaching NOT_ to the next TWO words
    text = re.sub(r"\b(not|no|never|didn't|don't|doesn't|isn't|aren't|wasn't|weren't)\s+(\w+)(\s+)?(\w+)?", 
                  lambda m: f"NOT_{m.group(2)} " + (f"NOT_{m.group(4)}" if m.group(4) else ""), 
                  text)
    # 4. Tokenize
    tokens = word_tokenize(text)
    # 5. Lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    
    # 6. Contrastive Weighting
    contrast_keywords = r"\b(but|however|although|though|yet|now)\b"
    if re.search(contrast_keywords, text, re.IGNORECASE):
        parts = re.split(contrast_keywords, text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 2:
            suffix_clean = [
                lemmatizer.lemmatize(w.lower(), pos='v') 
                for w in word_tokenize(parts[2].translate(str.maketrans('', '', string.punctuation)))
            ]
            # 5x Weighting for the shift
            for _ in range(4):
                cleaned_tokens.extend(suffix_clean)

    return " ".join(cleaned_tokens)

# --- Label Mapping ---
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# --- Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tfidf:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    raw_text = data['text']
    
    # 1. Preprocess
    clean_text = pre_process(raw_text)
    
    # 2. Vectorize
    text_vector = tfidf.transform([clean_text])
    
    # 3. Predict
    numeric_prediction = model.predict(text_vector)[0]
    emotion = LABEL_MAP.get(numeric_prediction, "neutral")
    
    # 4. Confidence (using predict_proba for LogisticRegression)
    probs = model.predict_proba(text_vector)[0]
    confidence = float(np.max(probs))

    return jsonify({
        "mood": emotion,
        "confidence": round(confidence, 4),
        "clean_text": clean_text
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model or not tfidf:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    raw_text = data['text']
    
    # 1. Local Prediction
    clean_text = pre_process(raw_text)
    text_vector = tfidf.transform([clean_text])
    numeric_prediction = model.predict(text_vector)[0]
    emotion = LABEL_MAP.get(numeric_prediction, "neutral")
    probs = model.predict_proba(text_vector)[0]
    confidence = float(np.max(probs))

    # 2. Groq AI Analysis
    ai_analysis = "Groq API Key not found. Please set it in the .env file for detailed AI insights."
    if groq_client:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert sentiment analyst specializing in English and Hinglish (Hindi + English). Analyze the following text.\n\n"
                                   "Provide your response in EXACTLY this format:\n"
                                   "Label: [sadness, joy, love, anger, fear, surprise]\n"
                                   "Confidence: [0-100]%\n"
                                   "Emoji: [one best matching emoji]\n"
                                   "Explanation: [brief 2-sentence explanation]\n\n"
                                   "Example:\n"
                                   "Label: joy\n"
                                   "Confidence: 95%\n"
                                   "Emoji: 😍\n"
                                   "Explanation: The user is expressing great happiness and satisfaction in Hinglish."
                    },
                    {
                        "role": "user",
                        "content": raw_text,
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=150,
            )
            raw_ai_response = chat_completion.choices[0].message.content.strip()
            
            # Parsing the structured response
            lines = raw_ai_response.split('\n')
            ai_data = {}
            for line in lines:
                if ':' in line:
                    k, v = line.split(':', 1)
                    ai_data[k.strip().lower()] = v.strip()
            
            # Extract Label
            if 'label' in ai_data:
                ai_label = ai_data['label'].lower().replace('.', '')
                if ai_label in LABEL_MAP.values():
                    emotion = ai_label
            
            # Extract Confidence
            if 'confidence' in ai_data:
                conf_match = re.search(r"(\d+)", ai_data['confidence'])
                if conf_match:
                    confidence = int(conf_match.group(1)) / 100.0
            
            # Extract Emoji
            ai_emoji = ai_data.get('emoji', None)
            
            # Extract Explanation
            if 'explanation' in ai_data:
                ai_analysis = ai_data['explanation']
            else:
                ai_analysis = raw_ai_response # Fallback


        except Exception as e:
            print(f"Groq API Error: {e}")
            ai_analysis = "Could not reach Groq AI at this time. Using local model only."

    return jsonify({
        "mood": emotion,
        "confidence": round(confidence, 4),
        "clean_text": clean_text,
        "ai_analysis": ai_analysis,
        "ai_emoji": ai_emoji if 'ai_emoji' in locals() else None
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Use PORT from environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
