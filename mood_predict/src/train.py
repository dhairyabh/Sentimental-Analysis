import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import pre_process_text

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'text.csv'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

def train_model():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Check if cleaned_text already exists or needs to be created
    print("Preprocessing text... (this may take a minute)")
    df['cleaned_text'] = df['text'].apply(pre_process_text)
    
    X = df['cleaned_text']
    y = df['label']
    
    label_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing data (Enhanced)...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print("Training optimized LinearSVC model...")
    # Best parameters found: C=0.1, class_weight='balanced'
    best_model = LinearSVC(C=0.1, class_weight='balanced', random_state=42, max_iter=3000)
    best_model.fit(X_train_tfidf, y_train)
    
    print("Evaluating model...")
    y_pred = best_model.predict(X_test_tfidf)
    
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map.keys())])
    print(report)
    
    # Save evaluation report to a file
    with open(os.path.join(MODELS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Parameters: C=0.1, class_weight='balanced', max_features=20000\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n\n")
        f.write(report)
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.values(),
                yticklabels=label_map.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Emotion Classification Confusion Matrix')
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(MODELS_DIR, 'confusion_matrix.png')}")
    
    # Save model and vectorizer
    print("Saving models...")
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'svm_model.pkl'))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    print("Done!")

if __name__ == "__main__":
    train_model()
