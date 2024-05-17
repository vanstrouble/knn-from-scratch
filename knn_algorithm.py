import numpy as np
import pandas as pd


from multiprocessing import Pool
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, f1_score


class KNN:
    def __init__(self, k=3) -> None:
        """Init method for KNN"""
        self.k = k

    def fit(self, X_train, y_train):
        """Fit function to lazy lerarning"""
        self.X_train = self._ensure_array(X_train)
        self.y_train = self._ensure_array(y_train)

    def predict(self, X_test, multi=False):
        """Predict function"""
        X_test = self._ensure_array(X_test)

        if multi:
            with Pool(processes=None) as pool:
                results = pool.map(self._compute_distances, X_test)
            return results
        else:
            y_pred = [self._compute_distances(x_test) for x_test in X_test]
            return np.array(y_pred)

    def _compute_distances(self, x):
        """Calculate distances between two points"""
        distances = np.linalg.norm(self.X_train - x, axis=1)
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
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError("Input data must be a numpy array.")


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_classes=2,
        flip_y=0.1,
        class_sep=0.5,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_neighbors = 3

    model = KNN(k=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test, False)

    # for index, element in enumerate(y_pred):
    #     print(f'{X_test[index]}, pred: {element}, expected: {y_test[index]}')

    print("")
    print(classification_report(y_test, y_pred))
    print("Accuracy score: ", round(accuracy_score(y_test, y_pred), 2))
    print("F1 score: ", round(f1_score(y_test, y_pred), 2))
