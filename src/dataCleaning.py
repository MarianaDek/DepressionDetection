import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def ensure_nltk_data():
    resources = {
        'punkt':       'tokenizers/punkt',
        'stopwords':   'corpora/stopwords',
        'wordnet':     'corpora/wordnet',
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

def preprocess_text(text):
    """
    1) Видаляє пунктуацію
    2) to lower-case
    3) токенізує
    4) видаляє стоп-слова
    5) лематизує
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in filtered)

def clean_data(csv_file):
    ensure_nltk_data()

    df = pd.read_csv(csv_file)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    texts = df['cleaned_text'].tolist()
    print("Data preprocessing complete. Cleaned texts stored in a Python list.")
    return texts



