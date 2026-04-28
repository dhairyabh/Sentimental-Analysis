import pandas as pd
import os
import random

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'text.csv')

def augment_sentiment_shifts():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    emotion_words = {
        0: ["sad", "depressed", "unhappy", "miserable", "lonely", "gloomy", "down"],
        1: ["happy", "joyful", "excited", "wonderful", "great", "excellent", "joyed", "joyous", "delighted"],
        2: ["loved", "loving", "fond", "passionate", "adorable"],
        3: ["angry", "mad", "furious", "annoyed", "frustrated", "irritated"],
        4: ["scared", "terrified", "fearful", "scary", "spooky", "frightened"],
        5: ["surprised", "shocked", "amazed", "astonished", "stunned"]
    }

    templates = [
        "i was {w1} but now i am {w2}",
        "i used to feel {w1} but right now i feel {w2}",
        "it was a {w1} day but then it became {w2}",
        "i felt so {w1} earlier however i am {w2} now",
        "even though it was originally {w1} i am actually {w2}",
        "i thought it was {w1} but i was wrong it is {w2}",
        "at first i was {w1} but then i felt {w2}",
        "{w1} context but {w2} result"
    ]

    new_data = []
    print("Generating 25,000 Shift-Aware samples...")
    
    # Generate ~25,000 samples
    for _ in range(25000):
        # Pick two different labels
        l1, l2 = random.sample(list(emotion_words.keys()), 2)
        w1 = random.choice(emotion_words[l1])
        w2 = random.choice(emotion_words[l2])
        
        template = random.choice(templates)
        text = template.format(w1=w1, w2=w2)
        
        # The label is ALWAYS the second (latest) emotion
        new_data.append([text, l2])

    new_df = pd.DataFrame(new_data, columns=['text', 'label'])
    combined_df = pd.concat([df, new_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
    
    print(f"Added {len(new_data)} shift samples. Final dataset size: {len(combined_df)}")
    combined_df.to_csv(DATA_PATH, index=False)
    print("Sentiment shift augmentation complete.")

if __name__ == "__main__":
    augment_sentiment_shifts()
