import torch
import torch.nn as nn
import json
import os
import sys

# Add src to path just in case
sys.path.append(os.path.dirname(__file__))

# Re-define or import the model class
# In a real project, we'd move the model class to a common models.py file
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

class BiLSTMPredictor:
    def __init__(self, model_dir='models'):
        MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', model_dir))
        MODEL_PATH = os.path.join(MODELS_DIR, 'bilstm_mood_model.pth')
        VOCAB_PATH = os.path.join(MODELS_DIR, 'bilstm_vocab.json')
        CONFIG_PATH = os.path.join(MODELS_DIR, 'bilstm_config.json')
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_bilstm.py first.")
            
        with open(CONFIG_PATH, 'r') as f:
            self.config = json.load(f)
            
        with open(VOCAB_PATH, 'r') as f:
            self.vocab = json.load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = BiLSTMClassifier(
            self.config['vocab_size'],
            self.config['embedding_dim'],
            self.config['hidden_dim'],
            self.config['output_dim'],
            self.config['n_layers'],
            self.config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.id2label = {int(k): v for k, v in self.config['id2label'].items()}
        self.max_len = self.config['max_len']

    def predict(self, text):
        tokens = text.lower().split()[:self.max_len]
        indexed = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        length = [len(indexed)]
        
        if length[0] == 0:
            return "unknown", 0.0
            
        padded = indexed + [0] * (self.max_len - len(indexed))
        tensor = torch.LongTensor(padded).unsqueeze(0).to(self.device)
        length_tensor = torch.LongTensor(length)
        
        with torch.no_grad():
            output = self.model(tensor, length_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
        label = self.id2label.get(prediction.item(), "unknown")
        return label, confidence.item()

def run_interactive_prediction():
    try:
        predictor = BiLSTMPredictor()
    except FileNotFoundError as e:
        print(e)
        return

    print("\n--- BiLSTM Mood Predictor ---")
    print("Enter 'quit' to exit.")
    
    while True:
        text = input("\nHow are you feeling? ")
        if text.lower() == 'quit':
            break
            
        label, confidence = predictor.predict(text)
        print(f"Predicted Mood: {label} ({confidence:.2%} confidence)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_input = " ".join(sys.argv[1:])
        try:
            predictor = BiLSTMPredictor()
            label, confidence = predictor.predict(text_input)
            print(f"Input: {text_input}")
            print(f"Predicted Mood: {label} ({confidence:.2%} confidence)")
        except FileNotFoundError as e:
            print(e)
    else:
        run_interactive_prediction()
