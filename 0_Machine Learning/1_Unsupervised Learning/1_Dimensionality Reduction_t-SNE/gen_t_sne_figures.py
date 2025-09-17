#!/usr/bin/env python3
"""Generate figures for the t-SNE dimensionality reduction tutorial."""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

FIGURE_LIST: List[str] = [
    "tsne_embeddings.png",
    "tsne_perplexity_curve.png",
]


def make_output_dir() -> str:
    """Create (if needed) and return the figures directory."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset with overlapping clusters."""

    points, labels = make_blobs(
        n_samples=1200,
        n_features=10,
        centers=5,
        cluster_std=[1.4, 1.0, 1.2, 1.1, 1.5],
        random_state=seed,
    )
    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    return points, labels


def run_tsne_perplexities(
    points: np.ndarray,
    perplexities: List[int],
    seed: int = 42,
) -> Tuple[Dict[int, np.ndarray], List[float]]:
    """Compute t-SNE embeddings and KL divergences for given perplexities."""

    embeddings: Dict[int, np.ndarray] = {}
    kl_divergences: List[float] = []

    for perp in perplexities:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            n_iter=2000,
            n_iter_without_progress=400,
            random_state=seed,
        )
        embedding = tsne.fit_transform(points)
        embeddings[perp] = embedding
        kl_divergences.append(tsne.kl_divergence_)

    return embeddings, kl_divergences


def plot_embeddings(
    embeddings: Dict[int, np.ndarray],
    labels: np.ndarray,
    perplexities: List[int],
    output_path: str,
) -> None:
    """Plot embeddings for each perplexity in a grid with shared color mapping."""

    n_cols = len(perplexities)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.6), sharex=False, sharey=False)

    if n_cols == 1:
        axes = [axes]

    for ax, perp in zip(axes, perplexities):
        emb = embeddings[perp]
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
        ax.set_title(f"perplexity = {perp}")
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # Add a single legend using class labels.
    handles, legend_labels = [], []
    for class_id in np.unique(labels):
        handles.append(
            axes[0].scatter([], [], c=scatter.cmap(class_id % scatter.cmap.N), s=30, edgecolor="k", linewidth=0.3)
        )
        legend_labels.append(f"Class {class_id}")
    axes[-1].legend(handles, legend_labels, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_perplexity_curve(
    perplexities: List[int],
    kl_divergences: List[float],
    output_path: str,
) -> None:
    """Plot KL divergence as a function of perplexity."""

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(perplexities, kl_divergences, marker="o", color="#1f77b4", linewidth=1.6)
    ax.set_title("KL Divergence vs. Perplexity")
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("KL divergence")
    ax.set_xticks(perplexities)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Create the figures required by the t-SNE tutorial."""

    figures_dir = make_output_dir()
    points, labels = synthesize_data()
    perplexities = [10, 30, 50]

    embeddings, kl_divergences = run_tsne_perplexities(points, perplexities)

    embeddings_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_embeddings(embeddings, labels, perplexities, embeddings_path)

    kl_curve_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_perplexity_curve(perplexities, kl_divergences, kl_curve_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()
