import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def plot_knn_pca(X_train, y_train, X_test, y_pred, ds_name):
    """Plot KNN classification with PCA on 2D data."""
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

    # Labels and title
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.show()


def plot_confusion_matrix(y_test, y_pred, labels):
    """Plot confusion matrix using seaborn heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()


def plot_f1_score(
        max_k, f1_scores,
        sklearn_f1_scores=None,
        plot_title="F1 Score",
        ):
    sns.set_context("notebook", rc={"lines.linewidth": 3})
    sns.set_style("ticks")

    plt.figure(figsize=(10, 6))

    palette = sns.color_palette("flare", 2)
    custom_color = palette[0]
    sklearn_color = palette[1]

    if sklearn_f1_scores is not None:
        plt.plot(
            range(1, max_k + 1),
            sklearn_f1_scores,
            color=sklearn_color,
            label="sklearn KNN",
            marker=".",
        )
    plt.plot(
        range(1, max_k + 1),
        f1_scores,
        color=custom_color,
        label="Custom KNN",
        marker=".",
        )

    plt.title(plot_title, fontsize=14, weight="bold")
    plt.xlabel("K neighbors", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.legend(title="Model", title_fontsize="12", fontsize="11")

    sns.despine()
    plt.show()


def plot_elbow(
        max_k,
        error_rates,
        sklearn_error_rates=None,
        plot_title="Elbow Method Comparison"
        ):
    sns.set_context("notebook", rc={"lines.linewidth": 3})
    sns.set_style("ticks")

    plt.figure(figsize=(10, 6))

    palette = sns.color_palette("BuPu", 2)
    custom_color = palette[0]
    sklearn_color = palette[1]

    if sklearn_error_rates is not None:
        plt.plot(
            range(1, max_k + 1),
            sklearn_error_rates,
            color=sklearn_color,
            label="sklearn KNN",
            marker=".",
        )
    plt.plot(
        range(1, max_k + 1),
        error_rates,
        color=custom_color,
        label="Custom KNN",
        marker="."
        )

    plt.title(plot_title, fontsize=14, weight="bold")
    plt.xlabel("K neighbors", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.legend(title="Model", title_fontsize="12", fontsize="11")

    sns.despine()
    plt.show()
