import unittest
import numpy as np
from scratch_tree import ScratchTree, gini


class TreeChecks(unittest.TestCase):
    def test_impurity(self):
        self.assertEqual(gini(np.array([1, 1, 1])), 0)
        self.assertEqual(gini(np.array([0, 0, 1, 1])), 0.5)
        self.assertEqual(gini(np.array([], dtype=int)), 0)

    def test_separable_data(self):
        X = np.array([[0], [0], [1], [1]])
        y = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(ScratchTree(1).fit(X, y).predict(X), y)

    def test_constant_features(self):
        tree = ScratchTree(2).fit([[1], [1], [1]], [0, 1, 1])
        np.testing.assert_array_equal(tree.predict([[1], [2]]), [1, 1])

    def test_stopping_rules(self):
        X, y = [[0], [1], [2]], [0, 1, 1]
        self.assertEqual(ScratchTree(0).fit(X, y).predict([[0]])[0], 1)
        self.assertEqual(ScratchTree(3, min_size=3).fit(X, y).predict([[0]])[0], 1)

    def test_xor_requires_two_levels(self):
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 0]
        np.testing.assert_array_equal(ScratchTree(2).fit(X, y).predict(X), y)
        self.assertFalse(np.array_equal(ScratchTree(1).fit(X, y).predict(X), y))

    def test_invalid_inputs(self):
        with self.assertRaises(RuntimeError):
            ScratchTree().predict([[0]])
        for X, y in [([], []), ([[float('nan')]], [1]), ([[1]], [1, 2]), ([[1]], [1.5])]:
            with self.assertRaises(ValueError):
                ScratchTree().fit(X, y)
        with self.assertRaises(ValueError):
            ScratchTree().fit([[0]], [0]).predict([[0, 1]])


if __name__ == '__main__':
    unittest.main()
