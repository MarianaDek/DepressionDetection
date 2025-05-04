# src/ml_algorithms/logistic_regression_scratch.py

import numpy as np

class LogisticRegressionScratch:
    """
    Реалізація логістичної регресії з L2-регуляризацією та One-vs-Rest для багатокласу.
    Параметри:
      learning_rate: крок градієнтного спуску
      n_iters: кількість ітерацій
      lambda_param: коефіцієнт L2-регуляризації
    """
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.01):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.classes_ = None
        self.W = None
        self.b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)


        for idx, c in enumerate(self.classes_):

            y_binary = np.where(y == c, 1, 0)
            w = np.zeros(n_features)
            b = 0.0


            for _ in range(self.n_iters):
                linear_model = X.dot(w) + b
                y_pred = self._sigmoid(linear_model)

                error = y_pred - y_binary

                dw = (X.T.dot(error) + self.lambda_param * w) / n_samples
                db = np.sum(error) / n_samples

                w -= self.lr * dw
                b -= self.lr * db

            self.W[idx, :] = w
            self.b[idx] = b

    def predict(self, X: np.ndarray) -> np.ndarray:

        linear = X.dot(self.W.T) + self.b
        probs = self._sigmoid(linear)

        class_idx = np.argmax(probs, axis=1)
        return self.classes_[class_idx]
