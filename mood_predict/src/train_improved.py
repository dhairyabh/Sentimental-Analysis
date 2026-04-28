import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from preprocess import pre_process_text

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'text.csv'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

def train_improved():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    if len(df) > 100000:
        print("Sampling 100,000 rows for faster experimentation...")
        df = df.sample(100000, random_state=42)
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(pre_process_text)
    
    X = df['cleaned_text']
    y = df['label']
    
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing data (Enhanced)...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # We will test two models: LinearSVC and LogisticRegression
    # Using f1_macro to give equal importance to minority classes
    
    models_to_test = [
        {
            'name': 'LinearSVC',
            'model': LinearSVC(random_state=42, max_iter=3000, class_weight='balanced'),
            'params': {'C': [0.1, 1]}
        },
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
            'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        }
    ]
    
    best_overall_f1 = 0
    best_overall_model = None
    best_overall_name = ""
    
    for item in models_to_test:
        print(f"\nTuning {item['name']}...")
        grid = GridSearchCV(
            item['model'],
            item['params'],
            cv=3,
            scoring='f1_macro',
            verbose=1,
            n_jobs=-1
        )
        grid.fit(X_train_tfidf, y_train)
        
        y_pred = grid.predict(X_test_tfidf)
        current_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"{item['name']} Best Macro F1: {current_f1:.4f}")
        print(f"Best Params: {grid.best_params_}")
        
        if current_f1 > best_overall_f1:
            best_overall_f1 = current_f1
            best_overall_model = grid.best_estimator_
            best_overall_name = item['name']
            best_overall_params = grid.best_params_

    print(f"\nWinner: {best_overall_name} with Macro F1: {best_overall_f1:.4f}")
    
    # Final Evaluation
    y_pred = best_overall_model.predict(X_test_tfidf)
    print("\nFinal Classification Report:")
    report = classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map.keys())])
    print(report)
    
    # Save improved model
    print("Saving improved model...")
    joblib.dump(best_overall_model, os.path.join(MODELS_DIR, 'improved_model.pkl'))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'improved_tfidf_vectorizer.pkl'))
    
    # Save evaluation report
    with open(os.path.join(MODELS_DIR, 'improved_classification_report.txt'), 'w') as f:
        f.write(f"Model: {best_overall_name}\n")
        f.write(f"Best Parameters: {best_overall_params}\n")
        f.write(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}\n\n")
        f.write(report)
        
    # Improved Confusion Matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_map.values(),
                yticklabels=label_map.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Improved Emotion Classification ({best_overall_name})')
    plt.savefig(os.path.join(MODELS_DIR, 'improved_confusion_matrix.png'))
    
    print("Done!")

if __name__ == "__main__":
    train_improved()
