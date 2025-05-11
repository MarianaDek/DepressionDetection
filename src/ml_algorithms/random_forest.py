import numpy as np
from numba import njit, prange
from joblib import Parallel, delayed

@njit
def gini(y):
    classes = np.unique(y)
    total = y.size
    res = 1.0
    for c in classes:
        cnt = 0
        for yi in y:
            if yi == c:
                cnt += 1
        p = cnt / total
        res -= p * p
    return res

@njit
def entropy(y):
    classes = np.unique(y)
    total = y.size
    res = 0.0
    for c in classes:
        cnt = 0
        for yi in y:
            if yi == c:
                cnt += 1
        p = cnt / total
        if p > 0.0:
            res -= p * np.log2(p)
    return res

@njit
def information_gain(parent_score, y, left_mask, right_mask, min_samples_leaf, criterion):
    y_l = y[left_mask]
    y_r = y[right_mask]
    if y_l.size < min_samples_leaf or y_r.size < min_samples_leaf:
        return -1e9
    score_l = gini(y_l) if criterion == 0 else entropy(y_l)
    score_r = gini(y_r) if criterion == 0 else entropy(y_r)
    n = y.size
    return parent_score - (y_l.size / n) * score_l - (y_r.size / n) * score_r

@njit(parallel=True)
def find_best_split_numba(X_col, y, thresholds, min_samples_leaf, parent_score, criterion):
    best_gain = -1e9
    best_thr = thresholds[0]
    for i in prange(thresholds.size):
        thr = thresholds[i]
        left_mask = X_col <= thr
        right_mask = ~left_mask
        gain = information_gain(parent_score, y, left_mask, right_mask, min_samples_leaf, criterion)
        if gain > best_gain:
            best_gain = gain
            best_thr = thr
    return best_gain, best_thr

class DecisionTree:
    class Node:
        __slots__ = ("feat", "thr", "left", "right", "value")
        def __init__(self, feat=None, thr=None, left=None, right=None, *, value=None):
            self.feat = feat
            self.thr = thr
            self.left = left
            self.right = right
            self.value = value

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        criterion="gini"
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = 0 if criterion == "gini" else 1
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or y.size < self.min_samples_split
            or np.unique(y).size == 1
        ):
            leaf_val = np.bincount(y).argmax()
            return DecisionTree.Node(value=leaf_val)

        parent_score = gini(y) if self.criterion == 0 else entropy(y)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(self.n_features_)))
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * self.n_features_))
        else:
            k = self.n_features_
        feats = np.random.choice(self.n_features_, k, replace=False)

        best_gain, best_feat, best_thr = -1e9, None, None
        for feat in feats:
            col = X[:, feat]
            thr = np.percentile(col, np.linspace(0, 100, 10))
            gain, thr0 = find_best_split_numba(
                col, y, thr,
                self.min_samples_leaf,
                parent_score,
                self.criterion
            )
            if gain > best_gain:
                best_gain, best_feat, best_thr = gain, feat, thr0

        if best_feat is None:
            leaf_val = np.bincount(y).argmax()
            return DecisionTree.Node(value=leaf_val)

        mask = X[:, best_feat] <= best_thr
        left = self._grow_tree(X[mask], y[mask], depth + 1)
        right = self._grow_tree(X[~mask], y[~mask], depth + 1)
        return DecisionTree.Node(best_feat, best_thr, left, right)

    def predict(self, X: np.ndarray):
        out = np.empty(X.shape[0], dtype=np.int64)
        for i, x in enumerate(X):
            node = self.root
            while node.value is None:
                if x[node.feat] <= node.thr:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.value
        return out

class RandomForestScratch:

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        criterion="gini",
        n_jobs=1,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trees = []
        self.n_classes_ = None

    def _build_one_tree(self, seed, X, y):
        np.random.seed(seed)
        if self.bootstrap:
            idxs = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_s, y_s = X[idxs], y[idxs]
        else:
            X_s, y_s = X, y
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion
        )
        tree.fit(X_s, y_s)
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.n_classes_ = np.unique(y).size
        seeds = np.random.randint(0, 100000, size=self.n_estimators)
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_one_tree)(seed, X, y) for seed in seeds
        )

    def predict_proba(self, X: np.ndarray):
        all_probs = []
        for tree in self.trees:
            preds = tree.predict(X)
            oh = np.zeros((preds.size, self.n_classes_), dtype=np.float64)
            for i, c in enumerate(preds):
                oh[i, c] = 1.0
            all_probs.append(oh)
        return np.mean(all_probs, axis=0)

    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
