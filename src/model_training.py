# src/model_training.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from dataVectorizations import vectorize_texts
from model_serialization import save_model

def load_labels(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df['target'].values

def train_and_save_model(
        texts: list[str],
        y: np.ndarray,
        classifier,
        algo_name: str,
        vectorizer_key: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        models_dir: str = "../models"
) -> dict:
    """
    1) Векторизує тексти за vectorizer_key у векторізаціях з dataVectorizations.
    2) Розбиває на train/test.
    3) Навчає переданий classifier.
    4) Оцінює accuracy + classification_report.
    5) Зберігає кортеж (classifier, vectorizer) у models_dir/algo_name_vectorizer_key.pkl

    Повертає словник з метриками та ім’ям файлу моделі.
    """
    # 1) Отримати векторизатор і матрицю ознак
    vecs = vectorize_texts(texts)
    vec_obj, X = vecs[vectorizer_key]

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    # 3) Навчання
    classifier.fit(
        X_train.toarray() if hasattr(X_train, "toarray") else X_train,
        y_train
    )

    # 4) Оцінка
    y_pred = classifier.predict(
        X_test.toarray() if hasattr(X_test, "toarray") else X_test
    )
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # 5) Збереження
    os.makedirs(models_dir, exist_ok=True)
    filename = f"{algo_name}_{vectorizer_key}.pkl"
    path = os.path.join(models_dir, filename)
    # зберігаємо tuple(classifier, vectorizer) для інференсу
    save_model((classifier, vec_obj), path)

    return {
        "algorithm": algo_name,
        "vectorizer": vectorizer_key,
        "accuracy": round(acc, 4),
        "report": report,
        "model_filepath": path
    }
