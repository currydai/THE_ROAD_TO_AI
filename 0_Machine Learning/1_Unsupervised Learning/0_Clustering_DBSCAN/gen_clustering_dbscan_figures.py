#!/usr/bin/env python3
"""Generate figures for the DBSCAN clustering tutorial.

This script creates synthetic two-dimensional data, applies DBSCAN, and
exports two visualizations used by the LaTeX tutorial files:
1. Cluster assignments over the synthetic data set.
2. The ordered k-distance plot for diagnosing a suitable epsilon.
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Figures produced by this script; referenced by the tutorial documents.
FIGURE_LIST = [
    "dbscan_clusters.png",
    "dbscan_k_distance.png",
]


def make_output_dir() -> str:
    """Return the absolute path of the figures directory, creating it if needed."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 7) -> np.ndarray:
    """Generate a noisy data set with three dense regions for clustering."""

    rng = np.random.RandomState(seed)

    # Concentric circle cluster with noise in radius.
    angles = rng.uniform(0, 2 * np.pi, size=150)
    radii = rng.normal(loc=1.0, scale=0.05, size=150)
    circle = np.stack((radii * np.cos(angles), radii * np.sin(angles)), axis=1)

    # Dense Gaussian blobs placed at different centers.
    blob_a = rng.normal(loc=(-1.5, 1.5), scale=0.1, size=(100, 2))
    blob_b = rng.normal(loc=(1.6, 1.3), scale=0.12, size=(110, 2))

    # Uniform background noise scattered around the perimeter.
    noise = rng.uniform(low=-3.0, high=3.0, size=(90, 2))

    points = np.vstack((circle, blob_a, blob_b, noise))
    return points


def run_dbscan(points: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, DBSCAN]:
    """Fit DBSCAN on the provided points and return labels plus the model."""

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    model.fit(points)
    return model.labels_, model


def plot_clusters(points: np.ndarray, labels: np.ndarray, eps: float, min_samples: int, output_path: str) -> None:
    """Plot the DBSCAN cluster assignments and save as a PNG figure."""

    unique_labels = sorted(set(labels))
    cluster_labels = [lab for lab in unique_labels if lab != -1]

    # Build a color map for clusters (excluding noise) and append gray for noise.
    color_map = plt.cm.get_cmap("tab10", max(len(cluster_labels), 1))
    colors = []
    for lab in labels:
        if lab == -1:
            colors.append((0.6, 0.6, 0.6))  # gray for noise
        else:
            colors.append(color_map(lab % color_map.N))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=20, edgecolor="k", linewidth=0.2)
    ax.set_title(f"DBSCAN Clusters (eps={eps:.2f}, minPts={min_samples})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal", adjustable="box")

    # Build legend entries manually to keep noise separate.
    handles = []
    labels_text = []
    for lab in cluster_labels:
        handles.append(ax.scatter([], [], color=color_map(lab % color_map.N), s=40, edgecolor="k", linewidth=0.3))
        labels_text.append(f"Cluster {lab}")
    handles.append(ax.scatter([], [], color=(0.6, 0.6, 0.6), s=40, edgecolor="k", linewidth=0.3))
    labels_text.append("Noise")
    ax.legend(handles, labels_text, loc="best", fontsize=8, framealpha=0.9)

    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_k_distance(points: np.ndarray, k: int, output_path: str) -> None:
    """Create the ordered k-distance plot used to pick epsilon."""

    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(points)

    distances, _ = neighbors.kneighbors(points)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(np.arange(1, len(k_distances) + 1), k_distances, color="#1f77b4", linewidth=1.6)
    ax.set_title(f"Ordered k-distance (k={k})")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel("Distance to k-th neighbor")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Entry point: generate figures and print their output paths."""

    figures_dir = make_output_dir()
    points = synthesize_data(seed=11)

    eps = 0.35
    min_samples = 5
    labels, _ = run_dbscan(points, eps=eps, min_samples=min_samples)

    cluster_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_clusters(points, labels, eps=eps, min_samples=min_samples, output_path=cluster_path)

    k_distance_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_k_distance(points, k=min_samples, output_path=k_distance_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()
