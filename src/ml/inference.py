import os
import joblib
import numpy as np
from src.ml.dataCleaning import preprocess_text
from src.ml.vectorizers import vectorize_one, vote_proba_rf

CLASS_NAMES = {
    0: 'Stress',
    1: 'Depression',
    2: 'Bipolar disorder',
    3: 'Personality disorder',
    4: 'Anxiety'
}

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEPRESSION_TYPE_MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = next(
    (
        os.path.join(DEPRESSION_TYPE_MODEL_DIR, fname)
        for fname in os.listdir(DEPRESSION_TYPE_MODEL_DIR)
        if fname.startswith('RF_Multi_tfidf') and fname.endswith('.pkl')
    ),
    None
)
model, vectorizer = joblib.load(MODEL_PATH)

DEPRESSED_CLASS_INDEX = 1

def predict_affinity(text: str) -> tuple[int, np.ndarray]:

    cleaned = preprocess_text(text)
    X_vec = vectorize_one(cleaned, vectorizer)

    if hasattr(model, 'predict_proba'):
        raw = model.predict_proba(X_vec)[0]
        if np.isscalar(raw) or raw.ndim == 0:
            p1 = float(raw)
            proba = np.array([1 - p1, p1])
        else:
            proba = np.array(raw)
    elif hasattr(model, 'trees'):
        proba1 = vote_proba_rf(model, X_vec, DEPRESSED_CLASS_INDEX)[0]
        proba = np.array([1 - proba1, proba1])
    else:
        pred = model.predict(X_vec)[0]
        proba = np.zeros(len(CLASS_NAMES))
        proba[pred] = 1.0

    label = int(np.argmax(proba))
    return label, proba
