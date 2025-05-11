import numpy as np

class SVMScratch:
    def __init__(
        self,
        lambda_param: float = 0.0001,
        epochs: int = 50,
        batch_size: int = 100,
        random_state: int | None = None
    ):
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.W: np.ndarray  # (n_classes, n_features)
        self.b: np.ndarray  # (n_classes,)
        self.classes_: np.ndarray
        self.W_avg: np.ndarray
        self.b_avg: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_idx[val] for val in y])

        self.W = np.zeros((n_classes, n_features), dtype=float)
        self.b = np.zeros(n_classes, dtype=float)
        self.W_avg = np.zeros_like(self.W)
        self.b_avg = np.zeros_like(self.b)
        total_updates = 0

        for epoch in range(1, self.epochs + 1):
            eta = 1.0 / (self.lambda_param * epoch)
            perm = rng.permutation(n_samples)
            X_sh = X[perm]
            y_sh = y_idx[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                Xb = X_sh[start:end]
                yb = y_sh[start:end]
                m = len(yb)

                scores = Xb @ self.W.T + self.b

                Y = np.where(yb[:, None] == np.arange(n_classes), 1, -1)
                margins = Y * scores
                mask = margins < 1

                grad_W = self.lambda_param * self.W
                grad_b = np.zeros_like(self.b)


                if mask.any():
                    labels_masked = Y[mask]
                    X_masked = Xb[mask.any(axis=1)]

                    for c in range(n_classes):
                        idx_c = mask[:, c]
                        if idx_c.any():
                            grad_W[c] -= (Y[idx_c, c][:, None] * Xb[idx_c]).sum(axis=0) / m
                            grad_b[c] -= Y[idx_c, c].sum() / m


                self.W -= eta * grad_W
                self.b -= eta * grad_b


                total_updates += 1
                self.W_avg += self.W
                self.b_avg += self.b

        self.W = self.W_avg / total_updates
        self.b = self.b_avg / total_updates

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W.T + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
