import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import os
import json
import joblib

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'text.csv'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'bilstm_mood_model.pth')
VOCAB_SAVE_PATH = os.path.join(MODELS_DIR, 'bilstm_vocab.json')
CONFIG_SAVE_PATH = os.path.join(MODELS_DIR, 'bilstm_config.json')

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
        # text: [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Concat the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        # hidden: [batch size, hid dim * num directions]
        return self.fc(hidden)

class MoodDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        tokens = text.split()[:self.max_len]
        indexed = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        length = len(indexed)
        
        # Padding
        padded = indexed + [0] * (self.max_len - len(indexed))
        
        return torch.LongTensor(padded), torch.tensor(label), torch.tensor(length)

def build_vocab(texts, max_size=20000):
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, count in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_bilstm():
    # Setup
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Sampling for efficient training
    sample_size = int(os.getenv('SAMPLE_SIZE', 50000))
    print(f"Sampling {sample_size} rows for training...")
    df = df.sample(min(len(df), sample_size), random_state=42)
    
    print("Building vocabulary...")
    vocab = build_vocab(df['text'])
    with open(VOCAB_SAVE_PATH, 'w') as f:
        json.dump(vocab, f)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    max_len = 64
    train_dataset = MoodDataset(train_df['text'].values, train_df['label'].values, vocab, max_len)
    val_dataset = MoodDataset(val_df['text'].values, val_df['label'].values, vocab, max_len)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model params
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 128
    output_dim = len(df['label'].unique())
    n_layers = 2
    dropout = 0.3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Initializing BiLSTM model on {device}...")
    model = BiLSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Save config
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'n_layers': n_layers,
        'dropout': dropout,
        'max_len': max_len,
        'id2label': {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}
    }
    with open(CONFIG_SAVE_PATH, 'w') as f:
        json.dump(config, f)
    
    # Early Stopping Initialization
    early_stopping = EarlyStopping(patience=3)
    
    # Training Loop
    epochs = int(os.getenv('EPOCHS', 20))  # Set via env or default to 20
    best_val_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for texts, labels, lengths in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(texts, lengths)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for texts, labels, lengths in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts, lengths)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                preds = predictions.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = epoch_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved!")
            
        # Check Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1} to prevent overfitting.")
            break

    print("Training complete!")

if __name__ == "__main__":
    train_bilstm()
