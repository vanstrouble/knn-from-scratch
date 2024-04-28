import numpy as np

# from multiprocessing import Pool
from collections import Counter


class KNN:
    def __init__(self, k=3) -> None:
        """_summary_

        Args:
            k (int, optional): K neighbors number. Defaults to 3.
        """
        self.k = k

    def fit(self, X_train, y_train):
        """Fit function to lazy lerarning

        Args:
            X_train (Pandas DataFrame): Training set data
            y_train (Pandas DataFrame): Labeled training set data
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = [self._compute_distances(x_test) for x_test in X_test]
        return y_pred

    def _compute_distances(self, x_test):
        distances = [np.linalg.norm(x_test - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common()[0][0]
        return most_common


if __name__ == "__main__":
    pass
