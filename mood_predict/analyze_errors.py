import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import pre_process_text

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), 'text.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

def analyze_errors():
    print("Loading data and models...")
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    
    # We need the same split as in train.py to get the test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("Preprocessing test data...")
    # Note: Using original text for display but cleaned for prediction
    X_test_cleaned = X_test.apply(pre_process_text)
    X_test_vectorized = tfidf.transform(X_test_cleaned)
    
    print("Predicting...")
    y_pred = model.predict(X_test_vectorized)
    
    # Analysis for Love (2)
    print("\n--- Analysis for 'Love' ---")
    love_mask = (y_test == 2)
    love_pred_mask = (y_pred == 2)
    
    # Confusion for Love
    cm = confusion_matrix(y_test, y_pred)
    love_idx = 2
    for i in range(6):
        if i != love_idx:
            print(f"True {label_map[i]} predicted as Love: {cm[i][love_idx]}")
    
    # Analysis for Surprise (5)
    print("\n--- Analysis for 'Surprise' ---")
    surprise_idx = 5
    for i in range(6):
        if i != surprise_idx:
            print(f"True {label_map[i]} predicted as Surprise: {cm[i][surprise_idx]}")

    # Display some examples of Joy being predicted as Love
    joy_as_love = np.where((y_test == 1) & (y_pred == 2))[0]
    print("\nTop 5 True Joy predicted as Love:")
    for idx in joy_as_love[:5]:
        print(f"- {X_test.iloc[idx]}")

    # Display some examples of Fear being predicted as Surprise
    # (Assuming Fear is often confused with Surprise)
    fear_as_surprise = np.where((y_test == 4) & (y_pred == 5))[0]
    print("\nTop 5 True Fear predicted as Surprise:")
    for idx in fear_as_surprise[:5]:
        print(f"- {X_test.iloc[idx]}")

if __name__ == "__main__":
    analyze_errors()
