import numpy as np

class DecisionTree:
    """
    Рекурсивна реалізація дерева рішень із контролем глибини та мінімальної кількості зразків.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        num_labels = len(np.unique(y))

        # Умови зупинки
        if (depth >= self.max_depth) or (num_labels == 1) or (n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTree.Node(value=leaf_value)

        # Знаходимо найкраще розщеплення
        best_feat, best_thresh = self._best_criteria(X, y)
        if best_feat is None:
            return DecisionTree.Node(value=self._most_common_label(y))

        # Розділення
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh
        if left_idxs.sum() < self.min_samples_leaf or right_idxs.sum() < self.min_samples_leaf:
            return DecisionTree.Node(value=self._most_common_label(y))

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        return DecisionTree.Node(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_criteria(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_gini = self._gini(y)

        for feat in range(self.n_features_):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idxs = y[X[:, feat] <= t]
                right_idxs = y[X[:, feat] > t]
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                gain = self._information_gain(y, left_idxs, right_idxs, parent_gini)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = t
        return split_idx, split_thresh

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs**2)

    def _information_gain(self, parent, l_child, r_child, parent_gini):
        num_parent = len(parent)
        num_l, num_r = len(l_child), len(r_child)
        gini_l, gini_r = self._gini(l_child), self._gini(r_child)
        weighted_gini = (num_l/num_parent)*gini_l + (num_r/num_parent)*gini_r
        return parent_gini - weighted_gini

    def _most_common_label(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X: np.ndarray):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForestScratch:
    """
    Random Forest із повноцінними деревами рішень.
    """
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []

        # Визначаємо кількість фіч для кожного дерева
        if isinstance(self.max_features, str) and self.max_features == "sqrt":
            feat_count = int(np.sqrt(n_features))
        elif isinstance(self.max_features, float):
            feat_count = int(self.max_features * n_features)
        else:
            feat_count = n_features

        for i in range(self.n_estimators):
            # Bootstrap-выборка
            if self.bootstrap:
                idxs = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
            else:
                X_sample, y_sample = X, y

            # Вибираємо підмножину фіч
            features = np.random.choice(n_features, feat_count, replace=False)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample[:, features], y_sample)
            tree.features = features
            self.trees.append(tree)

    def predict(self, X: np.ndarray):
        # Збираємо передбачення всіх дерев
        tree_preds = np.array([
            tree.predict(X[:, tree.features]) for tree in self.trees
        ])  # shape: (n_trees, n_samples)

        # Голосування
        y_pred = []
        for sample_preds in tree_preds.T:
            counts = np.bincount(sample_preds)
            y_pred.append(np.argmax(counts))
        return np.array(y_pred)

