import pandas as pd
import numpy as np
import os
import joblib
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mood_predict', 'src'))
from preprocess import pre_process_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Paths
CSV_PATH = os.path.join("mood_predict", "text.csv")
MODEL_DIR = os.path.join("mood_predict", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "production_model_v8.pkl")
VECTORIZER_SAVE_PATH = os.path.join(MODEL_DIR, "production_tfidf_v8.pkl")

def train():
    print(f"Loading augmented dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    print("Preprocessing text using fixed logic...")
    # Using the fixed logic that preserves NOT_ prefixes
    df['cleaned_text'] = df['text'].apply(pre_process_text)

    X = df['cleaned_text']
    y = df['label']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print("Vectorizing (Max Features: 100,000, Ngrams: 1-3)...")
    tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 3), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training LogisticRegression (balanced, optimized)...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=5000, n_jobs=-1, C=1.0)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_tfidf)
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    LABEL_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]))

    # Test final tricky cases before saving
    print("\nFinal Model Verification (Negations):")
    test_cases = ["i am not happy", "not sad at all", "this isn't a surprise", "i never feel joy"]
    test_cleaned = [pre_process_text(t) for t in test_cases]
    test_vecs = tfidf.transform(test_cleaned)
    test_preds = model.predict(test_vecs)
    for text, pred in zip(test_cases, test_preds):
        print(f"'{text}' -> {LABEL_MAP[pred].upper()}")

    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(tfidf, VECTORIZER_SAVE_PATH)
    print("Done! Version 3 production model suite is ready.")

if __name__ == "__main__":
    train()
