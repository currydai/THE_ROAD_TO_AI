#!/usr/bin/env python3
"""Generate figures for the K-means clustering tutorial."""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Figures produced by this script and referenced by the LaTeX documents.
FIGURE_LIST: List[str] = [
    "kmeans_clusters.png",
    "kmeans_elbow.png",
]


def make_output_dir() -> str:
    """Return the absolute path of the figures directory, creating it if needed."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 21) -> np.ndarray:
    """Generate three Gaussian blobs plus mild noise for clustering."""

    rng = np.random.RandomState(seed)

    blob_a = rng.normal(loc=(-1.5, 1.2), scale=0.18, size=(180, 2))
    blob_b = rng.normal(loc=(1.8, 1.6), scale=0.22, size=(200, 2))
    blob_c = rng.normal(loc=(0.0, -1.7), scale=0.2, size=(190, 2))
    noise = rng.normal(loc=(0.0, 0.0), scale=0.55, size=(60, 2))

    points = np.vstack((blob_a, blob_b, blob_c, noise))
    points -= points.mean(axis=0)
    points /= points.std(axis=0)
    return points


def run_kmeans(points: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, KMeans]:
    """Fit K-means on the provided points and return labels and the model."""

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    labels = model.fit_predict(points)
    return labels, model


def plot_clusters(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray, output_path: str) -> None:
    """Scatter plot of clustered points with centroid markers."""

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10",
                         s=25, alpha=0.85, edgecolor="k", linewidth=0.2)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", s=120,
               marker="X", linewidths=1.5, edgecolor="white", label="Centroid")
    ax.set_title("K-means Clusters (K=3)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_elbow(points: np.ndarray, output_path: str) -> None:
    """Compute and plot the inertia curve for a range of cluster counts."""

    ks = list(range(1, 9))
    inertias: List[float] = []

    for k in ks:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        model.fit(points)
        inertias.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(ks, inertias, marker="o", color="#1f77b4", linewidth=1.6)
    ax.set_title("Elbow Curve of Inertia")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Generate the tutorial figures and print their file paths."""

    figures_dir = make_output_dir()
    points = synthesize_data()

    labels, model = run_kmeans(points, n_clusters=3)
    cluster_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_clusters(points, labels, model.cluster_centers_, cluster_path)

    elbow_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_elbow(points, elbow_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()
