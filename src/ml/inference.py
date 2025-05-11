import os
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack

from src.ml.dataCleaning import preprocess_text
from src.ml.vectorizers import vectorize_one, vote_proba_rf

BINARY_CLASS_NAMES = {0: 'Not Depressed', 1: 'Depressed'}
TYPE_CLASS_NAMES   = {
    0: 'Stress',
    1: 'Depression',
    2: 'Bipolar disorder',
    3: 'Personality disorder',
    4: 'Anxiety'
}

MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'models')
)

# знаходимо файли
BIN_PATH = next(
    os.path.join(MODELS_DIR, f)
    for f in os.listdir(MODELS_DIR)
    if f.startswith('NB_Binary_count') and f.endswith('.pkl')
)
TYPE_PATH = next(
    os.path.join(MODELS_DIR, f)
    for f in os.listdir(MODELS_DIR)
    if f.startswith('NB_Multi_ngrams') and f.endswith('.pkl')
)

# завантажуємо
bin_model,  bin_vec  = joblib.load(BIN_PATH)
type_model, type_vec = joblib.load(TYPE_PATH)


def _combine_text_and_meta(X_text, age: float, gender: float, age_cat: float):
    """
    Прикріплює до X_text (1×D_text) вік, стать, категорію +
    дороблює нулі так, щоб збіглося із bin_model.n_features_in_.
    """
    # 1) перші 3 мета-ознаки
    meta = np.array([[age, gender, age_cat]])  # shape (1,3)

    # 2) порахуємо, скільки фіч чекає модель
    total_expected = bin_model.n_features_in_
    text_dim = X_text.shape[1]
    meta_dim = meta.shape[1]

    # 3) скільки ще «порожніх» фічів треба підсипати
    missing = total_expected - (text_dim + meta_dim)
    if missing < 0:
        raise ValueError(
            f"Model expects {total_expected} features, "
            f"but text+meta give {text_dim+meta_dim}"
        )

    # 4) побудуємо масив нулів shape=(1,missing)
    pad = np.zeros((1, missing), dtype=float)

    # 5) з'єднаємо залежно від того, sparse чи ні
    if hasattr(X_text, 'toarray'):
        return sparse_hstack([
            X_text,
            csr_matrix(meta),
            csr_matrix(pad)
        ])
    else:
        return np.hstack([X_text, meta, pad])


def predict_full(text: str, age: float, gender: float, age_cat: float):
    """
    1) бінарна модель (text+meta)
    2) якщо депресія – мультиклас по тексту
    """
    # векторизуємо текст
    cleaned = preprocess_text(text)
    X_txt   = vectorize_one(cleaned, bin_vec)    # (1×D_text)
    X_full  = _combine_text_and_meta(X_txt, age, gender, age_cat)

    # прогноз бінарної моделі
    if hasattr(bin_model, 'predict_proba'):
        p1 = bin_model.predict_proba(X_full)[0][1]
    else:
        p1 = float(bin_model.predict(X_full)[0] == 1)
    is_depr = p1 >= 0.5

    # якщо нема депресії
    if not is_depr:
        return False, p1, None, None

    # ---- мультикласова модель (тільки текст) ----
    X_type = vectorize_one(cleaned, type_vec)  # (1×D_type)
    if hasattr(type_model, 'predict_proba'):
        proba = type_model.predict_proba(X_type)[0]
    elif hasattr(type_model, 'trees'):
        votes = vote_proba_rf(type_model, X_type, class_index=1)
        base  = type_model.predict(X_type)[0]
        proba = np.zeros(len(TYPE_CLASS_NAMES))
        proba[base] = 1.0
    else:
        base  = type_model.predict(X_type)[0]
        proba = np.zeros(len(TYPE_CLASS_NAMES))
        proba[base] = 1.0

    idx = int(np.argmax(proba))
    return True, float(p1), TYPE_CLASS_NAMES[idx], np.array(proba)



