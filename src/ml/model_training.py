import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.naive_bayes import MultinomialNB

from src.ml.dataCleaning import preprocess_text
from src.ml.dataVectorizations import vectorize_texts
from src.ml.model_load_save import save_model
from src.ml.model_evaluation import evaluate_model


def load_dataset(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xls', '.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if 'label' in df.columns:
        df = df.dropna(subset=['label'])
    else:
        last_col = df.columns[-1]
        if df[last_col].dtype.kind in ('i', 'f'):
            df = df.dropna(subset=[last_col])

    if {'title','text'}.issubset(df.columns) and df.dtypes[-1] in (np.int64, np.int32):
        texts = df['text'].astype(str).tolist()
        y     = df.iloc[:, -1].astype(int).values
        meta_X = None

    elif {'text','label','Age','Gender','Age Category'}.issubset(df.columns):
        texts = df['text'].astype(str).tolist()
        y     = df['label'].astype(int).values

        age = df[['Age']].to_numpy(dtype=float)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat = ohe.fit_transform(df[['Gender','Age Category']])
        meta_X = np.hstack([age, cat])

    else:
        raise KeyError(f"Невідомий формат датасету: {path}")

    return texts, meta_X, y

def combine_text_and_meta(X_text, meta_X):

    if meta_X is None:
        return X_text
    if hasattr(X_text, 'toarray'):
        return sparse_hstack([X_text, csr_matrix(meta_X)])
    return np.hstack([X_text, meta_X])


def preprocess_and_vectorize(dataset_path: str):
    texts, meta_X, y = load_dataset(dataset_path)
    cleaned = [preprocess_text(t) for t in texts]
    vecs = vectorize_texts(cleaned)

    full_vecs = {}
    for key, (vec_obj, X_text) in vecs.items():
        X_full = combine_text_and_meta(X_text, meta_X)
        full_vecs[key] = (vec_obj, X_full)

    return full_vecs, y


def fit_and_evaluate(
    X_full, y, classifier,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    if hasattr(X_tr, 'toarray'):
        X_fit, X_val = X_tr.toarray(), X_te.toarray()
    else:
        X_fit, X_val = X_tr, X_te

    if isinstance(classifier, MultinomialNB):
        X_fit = np.clip(X_fit, a_min=0.0, a_max=None)
        X_val = np.clip(X_val, a_min=0.0, a_max=None)

    start = time.time()
    trained = classifier.fit(X_fit, y_tr)
    duration = time.time() - start
    model = trained if trained is not None else classifier

    print("\n--- Evaluation Report ---")
    evaluate_model(model, X_val, y_te)

    preds = model.predict(X_val)
    metrics = {
        'accuracy': round(accuracy_score(y_te, preds), 4),
        'f1_score': round(f1_score(y_te, preds, average='weighted', zero_division=0), 4),
        'precision': round(precision_score(y_te, preds, average='weighted', zero_division=0), 4),
        'recall': round(recall_score(y_te, preds, average='weighted', zero_division=0), 4),
        'training_time_sec': round(duration, 2)
    }
    return model, metrics


def train_and_save_model(
    dataset_path, classifier, algo_name, vectorizer_key,
    models_dir: str = "../models"
):

    full_vecs, y = preprocess_and_vectorize(dataset_path)
    if vectorizer_key not in full_vecs:
        raise KeyError(f"Vectorizer '{vectorizer_key}' not found in {list(full_vecs)}")
    vec_obj, X_full = full_vecs[vectorizer_key]

    model, metrics = fit_and_evaluate(X_full, y, classifier)

    os.makedirs(models_dir, exist_ok=True)
    fname = f"{algo_name}_{vectorizer_key}.pkl"
    path = os.path.join(models_dir, fname)
    save_model((model, vec_obj), path)

    return {
        'algorithm': algo_name,
        'vectorizer': vectorizer_key,
        'model_filepath': path,
        **metrics
    }

