import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def plot_tsne(features, labels, title="t-SNE Visualization"):
    print("Running t-SNE... (this may take a while)")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )

    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap='tab10',
        s=5
    )

    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def run_tsne_analysis(features, true_labels):
    # ===== 1️True Labels =====
    plot_tsne(features, true_labels, title="t-SNE (True Labels)")

    # ===== Cluster Labels =====
    print("Running KMeans for visualization...")
    kmeans = KMeans(n_clusters=10)
    cluster_labels = kmeans.fit_predict(features)

    plot_tsne(features, cluster_labels, title="t-SNE (Cluster Labels)")