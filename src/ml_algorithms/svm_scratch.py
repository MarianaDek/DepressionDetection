# src/ml_algorithms/svm_scratch.py

import numpy as np

class SVMScratch:
    """
    One-vs-Rest SVM з оптимізацією через Pegasos (стохастичний субградієнтний спуск).
    Підтримує багатокласову класифікацію.
    """
    def __init__(self, lambda_param=0.0001, epochs=50, batch_size=100):
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None  # форма (n_classes, n_features)
        self.b = None  # форма (n_classes,)
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # ініціалізуємо ваги для кожного класу
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        # Номер класу для кожного y
        class_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_idx[val] for val in y])

        # Pegasos оптимізація
        for epoch in range(1, self.epochs + 1):
            # тета - швидкість навчання
            eta = 1.0 / (self.lambda_param * epoch)
            # перемішуємо дані
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y_idx[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # обчислюємо субградієнти для кожного класу
                for i in range(n_classes):
                    # мітка +1 чи -1 для поточного класа
                    labels = np.where(y_batch == i, 1, -1)
                    # марґінальні умови
                    margins = labels * (X_batch.dot(self.W[i]) + self.b[i])

                    idx = np.where(margins < 1)[0]

                    grad_w = self.lambda_param * self.W[i]
                    grad_b = 0.0
                    if len(idx) > 0:
                        grad_w -= (labels[idx, None] * X_batch[idx]).sum(axis=0) / len(y_batch)
                        grad_b -= labels[idx].sum() / len(y_batch)

                    self.W[i] -= eta * grad_w
                    self.b[i] -= eta * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X.dot(self.W.T) + self.b
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

