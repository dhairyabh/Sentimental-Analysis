import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define paths
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_PATH = os.path.join(MODELS_DIR, 'distilbert-mood')

class DeepEmotionPredictor:
    def __init__(self):
        self.load_model()
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Deep learning model not found at {MODEL_PATH}. Please run src/train_deep.py first.")
        
        print("Loading fine-tuned DistilBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        return self.model.config.id2label[prediction]

def main():
    try:
        predictor = DeepEmotionPredictor()
        print("\n=== Deep Learning Mood Predictor ===")
        print("Type 'quit' to exit.")
        print("-" * 34)
        
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
