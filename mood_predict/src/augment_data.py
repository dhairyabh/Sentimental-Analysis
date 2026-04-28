import pandas as pd
import random
import os

# Define relative paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'text.csv')

def augment_negations():
    print("Loading original dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Emotion Label Map
    # 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    
    negations = [
        ("not", ""), ("no", ""), ("never", ""), ("isn't", ""), 
        ("wasn't", ""), ("don't", ""), ("didn't", ""), ("haven't", "")
    ]
    
    positive_words = ["happy", "joyful", "excited", "good", "great", "wonderful", "amazing", "content", "delighted"]
    negative_words = ["sad", "angry", "scared", "fearful", "terrible", "bad", "horrible", "miserable", "depressed"]
    love_words = ["love", "affectionate", "care", "adore", "passion"]
    surprise_words = ["surprised", "shocked", "amazed", "astonished"]
    
    new_data = []

    print("Generating negated samples (Super-Augmentation mode)...")
    # Add ~10,000 sets (each set adds ~5 samples)
    for i in range(10000):
        # 1. Not Happy -> Sadness (0) 
        neg = random.choice(["not", "never", "isn't", "wasn't", "hardly", "no"])
        pos = random.choice(positive_words)
        new_data.append([f"I am {neg} {pos} at all", 0]) 
        new_data.append([f"this makes me feel {neg} {pos}", 0])
        new_data.append([f"it is {neg} {pos} to be here", 0])
        
        # 2. Not Sad -> Joy (1)
        neg = random.choice(["not", "never", "isn't", "wasn't"])
        low = random.choice(negative_words)
        new_data.append([f"I am {neg} {low} anymore", 1]) 
        new_data.append([f"i feel {neg} {low} and i love it", 1])
        new_data.append([f"life is {neg} {low} today", 1])
        
        # 3. Not Angry -> Joy (1)
        new_data.append([f"I'm {neg} angry or mad", 1])
        new_data.append([f"no longer {low} and feel great", 1])
        
        # 4. Never love / Don't love -> Sadness (0) / Anger (3)
        neg = random.choice(["don't", "never", "didn't", "no"])
        lov = random.choice(love_words)
        new_data.append([f"I {neg} {lov} this place", 3])
        new_data.append([f"i have {neg} {lov} left", 0])

        # 5. Not Surprise -> Sadness (0)
        neg = random.choice(["not", "no", "never"])
        surp = random.choice(surprise_words)
        new_data.append([f"this is {neg} a {surp} anymore", 0])

    # Add samples for combined phrases
    new_data.append(["i am not happy i am very sad", 0])
    new_data.append(["not sad at all actually very happy", 1])
    new_data.append(["i don't love you i hate you", 3])
    new_data.append(["it wasn't a surprise at all", 0])

    new_df = pd.DataFrame(new_data, columns=['text', 'label'])
    
    # Shuffle and combine
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    print(f"Added {len(new_data)} samples. Total dataset size: {len(combined_df)}")
    combined_df.to_csv(DATA_PATH, index=False)
    print("Dataset successfully augmented and saved.")

if __name__ == "__main__":
    augment_negations()
