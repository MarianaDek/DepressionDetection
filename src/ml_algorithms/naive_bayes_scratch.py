# src/ml_algorithms/naive_bayes_scratch.py

import numpy as np

class NaiveBayesScratch:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.clip(X, a_min=0, a_max=None)

        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.class_log_prior_ = np.log(counts) - np.log(n_samples)

        feature_count = np.zeros((n_classes, n_features), dtype=float)
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_count[idx, :] = X_c.sum(axis=0)

        eps = 1e-9
        smoothed_fc = feature_count + self.alpha + eps
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True) + eps

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        jll = X.dot(self.feature_log_prob_.T) + self.class_log_prior_
        class_idx = np.argmax(jll, axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        jll = X.dot(self.feature_log_prob_.T) + self.class_log_prior_
        e_jll = np.exp(jll - np.max(jll, axis=1, keepdims=True))
        return e_jll / e_jll.sum(axis=1, keepdims=True)
