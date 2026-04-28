import pandas as pd
import os
import random

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'text.csv')

def correct_noise_and_balance():
    print("Loading augmented dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Relabeling Fear-inducing keywords (Aggressive mode)...")
    
    # 1. Broad Relabeling for high-signal fear words
    # We target common mislabeled fear words.
    fear_keywords = ["scary", "terrified", "frightened", "spooky", "creepy", "fearful"]
    
    mask = df['text'].str.contains('|'.join(fear_keywords), case=False, na=False) & \
           ~df['text'].str.contains(r"not|no|never|don't|isn't|wasn't", case=False, na=False)
           
    relabel_count = mask.sum()
    df.loc[mask, 'label'] = 4
    print(f"Relabeled {relabel_count} samples to Fear (4).")

    # 2. Heavy Augmentation for Fear (4) and Surprise (5)
    new_data = []
    
    fear_words = ["scary", "terrifying", "creepy", "spooky", "frightening", "horrified", "terrified", "scared to death"]
    fear_templates = [
        "this is absolutely {w}", "i am so {w} right now", 
        "it was a {w} experience", "i feel {w} thinking about it",
        "that's {w}!", "oh god it's {w}", "i'm {w}", "it's {w} to see this"
    ]
    
    for _ in range(10000):
        t = random.choice(fear_templates).format(w=random.choice(fear_words))
        new_data.append([t, 4])
        
    surprise_words = ["surprise", "shocking", "amazing", "astonishing", "unbelievable", "omg", "wow"]
    for _ in range(5000):
        t = "{w}! i can't believe it".format(w=random.choice(surprise_words))
        new_data.append([t, 5])

    new_df = pd.DataFrame(new_data, columns=['text', 'label'])
    combined_df = pd.concat([df, new_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
    
    print(f"Added {len(new_data)} balanced samples. Final dataset size: {len(combined_df)}")
    combined_df.to_csv(DATA_PATH, index=False)
    print("Noise correction and heavy balancing complete.")

if __name__ == "__main__":
    correct_noise_and_balance()
