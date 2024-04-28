import numpy as np
import pandas as pd

# from multiprocessing import Pool
from collections import Counter


class KNN:
    def __init__(self, k=3) -> None:
        """Init method for KNN"""
        self.k = k

    def fit(self, X_train, y_train):
        """Fit function to lazy lerarning"""
        self.X_train = self._ensure_array(X_train)
        self.y_train = self._ensure_array(y_train)

    def predict(self, X_test):
        """Predict function"""
        X_test = self._ensure_array(X_test)
        y_pred = [self._compute_distances(x_test) for x_test in X_test]
        return y_pred

    def _compute_distances(self, x):
        """Calculate distances between two points"""
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common()[0][0]
        return most_common

    def _ensure_array(self, data):
        """Ensure input data is converted to numpy array"""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")


if __name__ == "__main__":
    pass
