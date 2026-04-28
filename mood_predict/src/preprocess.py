import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

lemmatizer = WordNetLemmatizer()

def pre_process_text(text):
    """
    Cleans input text by:
    1. Lowercasing
    2. Removing punctuation (except spaces)
    3. Handling negations (attaching NOT_ prefix)
    4. Tokenizing
    5. Lemmatizing
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation BEFORE handling negations
    # This prevents the underscore in NOT_word from being stripped.
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Handle negations by attaching NOT_ to the next TWO words
    # This helps catch "not very happy" or "never feel joy"
    # Logic: Look for negation word, then skip a space, then flag word1, then (optional) skip space and flag word2
    text = re.sub(r"\b(not|no|never|didn't|don't|doesn't|isn't|aren't|wasn't|weren't)\s+(\w+)(\s+)?(\w+)?", 
                  lambda m: f"NOT_{m.group(2)} " + (f"NOT_{m.group(4)}" if m.group(4) else ""), 
                  text)
    
    # 4. Tokenize
    tokens = word_tokenize(text)
    
    # 5. Lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    
    # 6. Contrastive Weighting (Sentiment Shift Awareness)
    # If "but", "however", or "now" are present, we double the weight of words after them
    # We use the original 'text' variable for regex split
    contrast_keywords = r"\b(but|however|although|though|yet|now)\b"
    if re.search(contrast_keywords, text, re.IGNORECASE):
        parts = re.split(contrast_keywords, text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 2: # [prefix, keyword, suffix]
            # Lemmatize the suffix words and append them again
            suffix_clean = [
                lemmatizer.lemmatize(w.lower(), pos='v') 
                for w in word_tokenize(parts[2].translate(str.maketrans('', '', string.punctuation)))
            ]
            # 5x Weighting for the shift
            for _ in range(4):
                cleaned_tokens.extend(suffix_clean)

    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    # Test cases
    test_samples = [
        "I am not happy",
        "I am very happy!",
        "This is not a surprise.",
        "I never feel sad."
    ]
    print("Testing Preprocessing:")
    for sample in test_samples:
        print(f"Original: {sample}")
        print(f"Cleaned:  {pre_process_text(sample)}\n")
