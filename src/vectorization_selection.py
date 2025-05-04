# src/vectorization_selection.py

import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataCleaning import clean_data
from src.dataVectorizations import vectorize_texts
from src.model_training import load_labels
from src.model_evaluation import evaluate_model


def find_best_vectorization(
    classifier_factory,
    algo_name: str,
    vecs: dict,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
):
    """
    Перебирає всі методи векторизації, навчає новий екземпляр через factory,
    викликає evaluate_model для кожного варіанту, та повертає:
      - ключ найкращої векторизації
      - точність на тестовій вибірці
      - кортеж (натренована модель, векторизатор)

    Друкує classification_report та confusion matrix для кожного методу,
    а також підсумкові accuracy по всіх варіантах.
    """
    best_score = -np.inf
    best_key = None
    best_pair = None
    acc_results = {}

    for key, (vec_obj, X) in vecs.items():
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        # convert sparse->dense if needed
        X_train_arr = X_train.toarray() if hasattr(X_train, "toarray") else X_train
        X_test_arr  = X_test.toarray()  if hasattr(X_test,  "toarray") else X_test

        # створюємо новий екземпляр через factory і навчаємо
        clf = classifier_factory()
        clf.fit(X_train_arr, y_train)

        print(f"\n=== {algo_name} + {key} ===")
        evaluate_model(clf, X_test_arr, y_test)
        acc = np.mean(clf.predict(X_test_arr) == y_test)
        acc_results[key] = acc

        if acc > best_score:
            best_score = acc
            best_key = key
            best_pair = (clf, vec_obj)

    # підсумок по accuracy
    print("\nAccuracy per vectorization:")
    for vk, score in acc_results.items():
        print(f"  {vk:<8}: {score:.4f}")

    return best_key, best_score, best_pair

