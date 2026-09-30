from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    prediction: int
    feature: int | None = None
    threshold: float | None = None
    left: 'Node | None' = None
    right: 'Node | None' = None


def gini(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(1.0 - np.sum((counts / counts.sum()) ** 2))


class ScratchTree:
    def __init__(self, max_depth=2, min_size=1):
        if max_depth < 0 or min_size < 1:
            raise ValueError('Depth must be nonnegative and min_size must be positive')
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError('X must be a nonempty two-dimensional array')
        if y.ndim != 1 or len(y) != len(X) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError('y must contain one integer class per row')
        if not np.isfinite(X).all():
            raise ValueError('Features must be finite')
        self.n_features = X.shape[1]
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        labels, counts = np.unique(y, return_counts=True)
        node = Node(int(labels[np.argmax(counts)]))
        if depth >= self.max_depth or len(y) <= self.min_size or len(labels) == 1:
            return node
        best_score = float('inf')
        best = None
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                mask = X[:, feature] < threshold
                n_left = int(mask.sum())
                if n_left == 0 or n_left == len(y):
                    continue
                score = (n_left * gini(y[mask]) + (len(y) - n_left) * gini(y[~mask])) / len(y)
                if score < best_score - 1e-12:
                    best_score = score
                    best = feature, float(threshold), mask
        if best is None:
            return node
        node.feature, node.threshold, mask = best
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        if self.root is None:
            raise RuntimeError('Fit the tree before predicting')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features or not np.isfinite(X).all():
            raise ValueError('Prediction features must match training and be finite')
        predictions = []
        for row in X:
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] < node.threshold else node.right
            predictions.append(node.prediction)
        return np.asarray(predictions, dtype=int)
