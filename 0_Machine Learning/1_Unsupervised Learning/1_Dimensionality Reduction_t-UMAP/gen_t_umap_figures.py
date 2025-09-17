#!/usr/bin/env python3
"""Generate figures for the UMAP dimensionality reduction tutorial.

The script creates a synthetic dataset, computes UMAP embeddings for a range
of neighbor counts, and exports:
1. A grid of embeddings coloured by class label.
2. A trustworthiness-versus-neighbors curve to quantify neighborhood
   preservation.

Run this file before compiling the accompanying LaTeX documents so that the
figures exist under igures/.
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as exc:  # pragma: no cover - dependency is external
    raise SystemExit(
        "The 'umap-learn' package is required. Install it with 'pip install umap-learn'."
    ) from exc

FIGURE_LIST: List[str] = [
    "umap_embeddings.png",
    "umap_neighbor_curve.png",
]


def make_output_dir() -> str:
    """Return the absolute path of the figures directory, creating it if needed."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 17) -> Tuple[np.ndarray, np.ndarray]:
    """Generate standardized sample points with partially overlapping clusters."""

    points, labels = make_blobs(
        n_samples=1400,
        n_features=12,
        centers=6,
        cluster_std=[1.2, 1.0, 1.3, 1.1, 1.4, 1.15],
        random_state=seed,
    )
    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    return points, labels


def run_umap(
    points: np.ndarray,
    neighbors_list: List[int],
    seed: int = 42,
) -> Tuple[Dict[int, np.ndarray], List[float]]:
    """Fit UMAP for each neighbor count and compute trustworthiness scores."""

    embeddings: Dict[int, np.ndarray] = {}
    trust_scores: List[float] = []

    for n_neighbors in neighbors_list:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            init="spectral",
            random_state=seed,
        )
        embedding = reducer.fit_transform(points)
        embeddings[n_neighbors] = embedding
        trust_scores.append(trustworthiness(points, embedding, n_neighbors=15))

    return embeddings, trust_scores


def plot_embeddings(
    embeddings: Dict[int, np.ndarray],
    labels: np.ndarray,
    neighbors_list: List[int],
    output_path: str,
) -> None:
    """Render a grid of embeddings corresponding to each neighbor setting."""

    n_cols = len(neighbors_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.6), sharex=False, sharey=False)

    if n_cols == 1:
        axes = [axes]

    for ax, n_neighbors in zip(axes, neighbors_list):
        emb = embeddings[n_neighbors]
        scatter = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=labels,
            cmap="tab10",
            s=22,
            alpha=0.85,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(f"n_neighbors = {n_neighbors}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # Construct a legend with one handle per class.
    handles, legend_labels = [], []
    for class_id in np.unique(labels):
        handles.append(
            axes[0].scatter([], [], c=scatter.cmap(class_id % scatter.cmap.N), s=32, edgecolor="k", linewidth=0.3)
        )
        legend_labels.append(f"Class {class_id}")
    axes[-1].legend(handles, legend_labels, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trustworthiness_curve(
    neighbors_list: List[int],
    trust_scores: List[float],
    output_path: str,
) -> None:
    """Plot trustworthiness against the number of neighbors."""

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(neighbors_list, trust_scores, marker="o", color="#1f77b4", linewidth=1.6)
    ax.set_title("Trustworthiness vs. Number of Neighbors")
    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("Trustworthiness (k=15)")
    ax.set_xticks(neighbors_list)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Generate all tutorial figures and report their file paths."""

    figures_dir = make_output_dir()
    points, labels = synthesize_data()
    neighbors_list = [10, 30, 50]

    embeddings, trust_scores = run_umap(points, neighbors_list)

    embeddings_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_embeddings(embeddings, labels, neighbors_list, embeddings_path)

    trust_curve_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_trustworthiness_curve(neighbors_list, trust_scores, trust_curve_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()