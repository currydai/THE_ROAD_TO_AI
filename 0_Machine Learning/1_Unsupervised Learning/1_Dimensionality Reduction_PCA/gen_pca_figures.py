#!/usr/bin/env python3
"""Generate figures for the PCA tutorial."""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURE_LIST: List[str] = [
    "pca_projection.png",
    "pca_explained_variance.png",
]


def make_output_dir() -> str:
    """Return the path to the figures directory, creating it if needed."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated 3D data with class labels for visualization."""

    rng = np.random.RandomState(seed)

    centers = np.array([
        [-2.5, 1.8, 0.5],
        [0.0, -2.0, -1.5],
        [2.2, 2.4, 1.0],
    ])
    covariances = [
        np.array([[0.6, 0.35, 0.1], [0.35, 0.7, 0.12], [0.1, 0.12, 0.4]]),
        np.array([[0.5, -0.25, 0.05], [-0.25, 0.6, -0.08], [0.05, -0.08, 0.3]]),
        np.array([[0.4, 0.18, -0.04], [0.18, 0.5, 0.06], [-0.04, 0.06, 0.35]]),
    ]
    counts = [180, 190, 200]

    points = []
    labels = []
    for idx, (center, cov, count) in enumerate(zip(centers, covariances, counts)):
        samples = rng.multivariate_normal(mean=center, cov=cov, size=count)
        points.append(samples)
        labels.append(np.full(count, idx, dtype=int))

    points = np.vstack(points)
    labels = np.concatenate(labels)

    # Standardize features to zero mean and unit variance.
    points -= points.mean(axis=0)
    points /= points.std(axis=0)
    return points, labels


def plot_projection(points: np.ndarray, labels: np.ndarray, pca: PCA, output_path: str) -> None:
    """Plot the first two principal components colored by class."""

    projected = pca.transform(points)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=labels,
        cmap="tab10",
        s=26,
        edgecolor="k",
        linewidth=0.25,
        alpha=0.85,
    )
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("Projection onto First Two Principal Components")
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.axvline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    handles, legend_labels = [], []
    for class_id in np.unique(labels):
        handles.append(
            ax.scatter([], [], c=scatter.cmap(class_id % scatter.cmap.N), s=40, edgecolor="k", linewidth=0.3)
        )
        legend_labels.append(f"Class {class_id}")
    ax.legend(handles, legend_labels, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_explained_variance(pca: PCA, output_path: str) -> None:
    """Plot explained variance ratio and cumulative curve."""

    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar(components, ratios, color="#1f77b4", alpha=0.7, label="Variance ratio")
    ax.plot(components, cumulative, marker="o", color="#ff7f0e", linewidth=1.6, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance")
    ax.set_title("Explained Variance by Principal Components")
    ax.set_xticks(components)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Generate PCA tutorial figures and print their file paths."""

    figures_dir = make_output_dir()
    points, labels = synthesize_data()

    pca = PCA(n_components=3, whiten=False, random_state=7)
    pca.fit(points)

    projection_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_projection(points, labels, pca, projection_path)

    variance_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_explained_variance(pca, variance_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()
