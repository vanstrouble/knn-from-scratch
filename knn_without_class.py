import time
import numpy as np
import multiprocessing as mp
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def knn_predict(X_train, y_train, x, k):
    distances = [np.linalg.norm(x - x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    return Counter(k_labels).most_common()[0][0]

def mp_knn_predict(args):
    x, X_train, y_train, k = args
    return knn_predict(X_train, y_train, x, k)

def evaluate(predictions, y_test):
    accuracy = np.sum(predictions == y_test) / len(y_test)
    # for index, element in enumerate(predictions):
    #     print(f'{X_test[index]}, pred: {element}, expected: {y_test[index]}')
    return np.around(accuracy, 2)


if __name__ == "__main__":
    # Cargar el dataset de Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k = 5
    cpu = mp.cpu_count()

    # KNN sin multiprocesamiento
    start = time.time()
    normal_pred = [knn_predict(X_train, y_train, x, k) for x in X_test]
    end = time.time()
    print("\nKNN sin multiprocesamiento")
    print(f"Accuracy: {evaluate(normal_pred, y_test)}, time: {end - start}")

    # KNN con multiprocesamiento
    start = time.time()
    pool = mp.Pool(cpu)
    mp_pred = pool.map(mp_knn_predict, [(x, X_train, y_train, k) for x in X_test])
    pool.close()
    pool.join()
    end = time.time()
    print("\nKNN con multiprocesamiento")
    print(f"Accuracy: {evaluate(mp_pred, y_test)}, time: {end - start}")
