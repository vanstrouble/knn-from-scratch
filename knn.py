import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from collections import Counter


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
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")


if __name__ == "__main__":
    a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=100)
    a_label = np.ones(len(a))

    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=50)
    b_label = 2 * np.ones(len(b))

    c = np.random.multivariate_normal([5, 18], [[1, 0], [0, 1]], size=70)
    c_label = 3 * np.ones(len(c))

    d = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], size=30)
    d_label = 4 * np.ones(len(d))

    X = np.concatenate((a, b, c, d))
    y = np.concatenate((a_label, b_label, c_label, d_label)).astype(int)

    colors = ['red', 'green', 'blue', 'orange']

    # plt.figure(figsize=(8, 6))

    for i in range(1, 5):

        points = X[y == i]

    #     plt.scatter(points[:, 0], points[:, 1], color=colors[i-1], label=f'Conjunto {chr(96+i)} (etiqueta {i})')

    # plt.title('Labeled data sets')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = KNN(k=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, label='Training Points')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=predictions, label='Test Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Test Points')
    plt.legend()
    plt.grid(True)
    plt.show()
