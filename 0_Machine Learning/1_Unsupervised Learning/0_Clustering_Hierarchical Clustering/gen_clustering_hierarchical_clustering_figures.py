#!/usr/bin/env python3
"""Generate figures for the hierarchical clustering tutorial."""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Figures referenced by the LaTeX tutorial files.
FIGURE_LIST: List[str] = [
    "hierarchical_dendrogram.png",
    "hierarchical_clusters.png",
]


def make_output_dir() -> str:
    """Return the path to the figures directory, creating it when missing."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 17) -> np.ndarray:
    """Create synthetic blobs with slight overlap for clustering demos."""

    rng = np.random.RandomState(seed)

    blob_a = rng.normal(loc=(-1.8, 1.4), scale=0.2, size=(160, 2))
    blob_b = rng.normal(loc=(1.9, 1.6), scale=0.24, size=(180, 2))
    blob_c = rng.normal(loc=(0.1, -1.6), scale=0.22, size=(170, 2))
    noise = rng.normal(loc=(0.0, 0.0), scale=0.55, size=(70, 2))

    points = np.vstack((blob_a, blob_b, blob_c, noise))
    points -= points.mean(axis=0)
    points /= points.std(axis=0)
    return points


def compute_hierarchy(points: np.ndarray) -> np.ndarray:
    """Return the linkage matrix using Ward's method."""

    return linkage(points, method="ward")


def plot_dendrogram(linkage_matrix: np.ndarray, output_path: str, truncate_level: int = 20) -> None:
    """Render and save a dendrogram from the linkage matrix."""

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=truncate_level,
        color_threshold=None,
        ax=ax,
    )
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)")
    ax.set_xlabel("Merged nodes")
    ax.set_ylabel("Distance")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cluster_scatter(points: np.ndarray, linkage_matrix: np.ndarray, output_path: str, n_clusters: int = 3) -> None:
    """Cut the dendrogram to obtain cluster labels and plot a scatter chart."""

    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="tab10",
        s=28,
        edgecolor="k",
        linewidth=0.2,
        alpha=0.85,
    )
    ax.set_title(f"Hierarchical Clusters (Ward, K={n_clusters})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # Build a legend using unique cluster identifiers.
    handles, legend_labels = [], []
    for cluster_id in np.unique(labels):
        handles.append(ax.scatter([], [], c=scatter.cmap(cluster_id % scatter.cmap.N), s=40, edgecolor="k", linewidth=0.3))
        legend_labels.append(f"Cluster {cluster_id}")
    ax.legend(handles, legend_labels, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Entry point: generate both hierarchical clustering figures."""

    figures_dir = make_output_dir()
    points = synthesize_data()
    linkage_matrix = compute_hierarchy(points)

    dendrogram_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_dendrogram(linkage_matrix, dendrogram_path)

    cluster_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_cluster_scatter(points, linkage_matrix, cluster_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()


