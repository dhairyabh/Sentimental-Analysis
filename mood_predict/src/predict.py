import joblib
import os
import sys
# Add src to path if running from this directory
sys.path.append(os.path.dirname(__file__))
from preprocess import pre_process_text

# Define paths
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_PATH = os.path.join(MODELS_DIR, 'production_model_v6.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'production_tfidf_v6.pkl')

class EmotionPredictor:
    def __init__(self):
        self.label_map = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        self.load_models()

    def load_models(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Model files not found. Please run src/train.py first.")
        
        self.model = joblib.load(MODEL_PATH)
        self.tfidf = joblib.load(VECTORIZER_PATH)

    def predict(self, text):
        cleaned_text = pre_process_text(text)
        vectorized_text = self.tfidf.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)[0]
        return self.label_map[prediction]

def main():
    try:
        predictor = EmotionPredictor()
        print("\n=== Mood Predictor ===")
        print("Type 'quit' to exit.")
        print("-" * 23)
        
        while True:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                continue
                
            prediction = predictor.predict(user_input)
            print(f"Predicted Emotion: {prediction.upper()}")
            
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
