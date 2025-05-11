# src/ml/vectorizers.py

import numpy as np

def vectorize_one(cleaned_text: str, vec_obj):
    tokens = cleaned_text.split()

    if hasattr(vec_obj, 'transform'):
        return vec_obj.transform([cleaned_text])

    first = next(iter(vec_obj.values()))
    if isinstance(first, int):
        V = max(vec_obj.values()) + 1
        X = np.zeros((1, V), dtype=float)
        for t in tokens:
            idx = vec_obj.get(t)
            if idx is not None:
                X[0, idx] += 1
        return X

    if isinstance(first, np.ndarray):
        vecs = [vec_obj[t] for t in tokens if t in vec_obj]
        if vecs:
            return np.mean(vecs, axis=0).reshape(1, -1)
        else:
            return np.zeros((1, first.shape[0]), dtype=float)

    raise TypeError(f"Unsupported vectorizer type: {type(vec_obj)}")


def vote_proba_rf(rf_model, X: np.ndarray, class_index: int) -> np.ndarray:
    votes = np.stack([
        tree.predict(X[:, tree.features])
        for tree in rf_model.trees
    ], axis=0)
    return np.mean(votes == class_index, axis=0)
