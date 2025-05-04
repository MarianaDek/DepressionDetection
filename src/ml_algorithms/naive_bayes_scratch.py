# src/ml_algorithms/naive_bayes_scratch.py

import numpy as np

class NaiveBayesScratch:
    """
    Проста реалізація Multinomial Naive Bayes з Laplace smoothing.
    Для багатокласового класифікатора.

    Параметри:
      alpha — параметр згладжування (Laplace)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None  # log P(c)
        self.feature_log_prob_ = None  # log P(x_i | c)
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):


        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Обчислимо лог-пріори: log P(c) = log(n_c) - log(n_samples)
        self.class_log_prior_ = np.log(counts) - np.log(n_samples)

        # Обчислимо ймовірності ознак P(x_i|c)
        # таблиця частот: N_{c,i} = sum_{j: y_j=c} X_{j,i}
        feature_count = np.zeros((n_classes, n_features), dtype=float)
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            # Підсумувати по рядках
            feature_count[idx, :] = X_c.sum(axis=0)

        # Застосувати Laplace smoothing: + alpha
        smoothed_fc = feature_count + self.alpha
        # Нормувати по сумі по ознаках
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Передбачення міток для X.
        """
        # Обчислити log P(c) + sum_i X_i * log P(x_i|c)
        jll = X.dot(self.feature_log_prob_.T) + self.class_log_prior_
        # Вибрати клас з максимальною сумою лог-імовірностей
        class_idx = np.argmax(jll, axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Повертає ймовірності кожного класу для кожного зразка.
        """
        jll = X.dot(self.feature_log_prob_.T) + self.class_log_prior_
        # Розгорнути в ймовірності через softmax
        e_jll = np.exp(jll - np.max(jll, axis=1, keepdims=True))
        return e_jll / e_jll.sum(axis=1, keepdims=True)
