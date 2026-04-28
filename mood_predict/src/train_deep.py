import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'text.csv'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, 'distilbert-mood')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1}

def train_deep_model():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # For CPU training or limited resources, sample the data
    # 500 rows is suitable for demonstration on a CPU.
    print("Sampling 500 rows for training...")
    df_sampled = df.sample(500, random_state=42)
    
    # Prepare datasets
    train_df, test_df = train_test_split(df_sampled, test_size=0.2, random_state=42, stratify=df_sampled['label'])
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=6,
        id2label={0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"},
        label2id={"sadness":0, "joy":1, "love":2, "anger":3, "fear":4, "surprise":5}
    )
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        push_to_hub=False,
        report_to="none",
        logging_steps=10
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("Training model (this will take time on CPU)...")
    trainer.train()
    
    print(f"Saving best model to {MODEL_OUTPUT_DIR}...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    train_deep_model()
