import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_knn_pca(
    X_train, y_train, X_test, y_pred,
    ds_name
):
    title = "KNN Classification with PCA on " + ds_name

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    # Create figure
    plt.figure()

    # Plot training points
    scatter_train = plt.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        cmap="viridis",
        marker="o",
        edgecolor="k",
        alpha=0.7,
        label="Training data",
    )

    # Plot test predictions
    scatter_test = plt.scatter(
        X_test_2d[:, 0],
        X_test_2d[:, 1],
        c=y_pred,
        cmap="coolwarm",
        marker="s",
        edgecolor="k",
        label="Test predictions",
    )

    # Create legend
    legend1 = plt.legend(
        *scatter_train.legend_elements(),
        title="Training Classes"
    )
    plt.gca().add_artist(legend1)
    plt.legend(
        *scatter_test.legend_elements(),
        title="Test Predictions",
        loc="upper right"
    )

    # Etiquetas y título
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.show()
