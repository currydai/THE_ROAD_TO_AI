#!/usr/bin/env python3
"""Generate figures for the Independent Component Analysis (ICA) tutorial."""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA

# Figures referenced in the LaTeX tutorial.
FIGURE_LIST: List[str] = [
    "ica_sources_vs_recovered.png",
    "ica_mixing_matrices.png",
]


def make_output_dir() -> str:
    """Return the figures directory, creating it when needed."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_sources(n_samples: int = 1500, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic independent source signals and a random mixing matrix."""

    rng = np.random.RandomState(seed)
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # periodic signal
    s2 = np.sign(np.sin(3 * time))  # square wave
    s3 = rng.laplace(loc=0.0, scale=1.0, size=n_samples)  # heavy-tailed noise

    sources = np.vstack((s1, s2, s3))
    sources /= sources.std(axis=1, keepdims=True)

    mixing = rng.normal(size=(3, 3))
    mixed = mixing @ sources
    mixed = mixed.T  # shape (n_samples, n_signals)

    return mixed, sources, mixing


def run_ica(mixed_signals: np.ndarray, n_components: int = 3, seed: int = 0) -> FastICA:
    """Fit FastICA to the mixed signals and return the fitted estimator."""

    ica = FastICA(
        n_components=n_components,
        whiten='unit-variance',
        max_iter=1000,
        tol=1e-4,
        random_state=seed,
    )
    ica.fit(mixed_signals)
    return ica


def plot_sources_vs_recovered(
    sources: np.ndarray,
    mixed: np.ndarray,
    recovered: np.ndarray,
    time: np.ndarray,
    output_path: str,
) -> None:
    """Compare original sources, mixed observations, and ICA recovered signals."""

    labels = ["Source 1", "Source 2", "Source 3"]
    fig, axes = plt.subplots(3, 3, figsize=(10.0, 6.0), sharex=True)

    for idx in range(3):
        axes[0, idx].plot(time, sources[idx], color="#1f77b4", linewidth=1.0)
        axes[0, idx].set_title(labels[idx])
        axes[0, idx].set_ylabel("Original")
        axes[0, idx].grid(alpha=0.2, linestyle="--", linewidth=0.4)

        axes[1, idx].plot(time, mixed[:, idx], color="#ff7f0e", linewidth=1.0)
        axes[1, idx].set_ylabel("Mixed")
        axes[1, idx].grid(alpha=0.2, linestyle="--", linewidth=0.4)

        axes[2, idx].plot(time, recovered[:, idx], color="#2ca02c", linewidth=1.0)
        axes[2, idx].set_ylabel("Recovered")
        axes[2, idx].grid(alpha=0.2, linestyle="--", linewidth=0.4)

    axes[2, 0].set_xlabel("Time")
    axes[2, 1].set_xlabel("Time")
    axes[2, 2].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mixing_matrices(
    mixing: np.ndarray,
    unmixing: np.ndarray,
    output_path: str,
) -> None:
    """Visualize the true mixing matrix and the estimated unmixing matrix."""

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    im0 = axes[0].imshow(mixing, cmap="coolwarm", aspect="auto")
    axes[0].set_title("True mixing matrix")
    axes[0].set_xlabel("Source")
    axes[0].set_ylabel("Observation")

    im1 = axes[1].imshow(unmixing, cmap="coolwarm", aspect="auto")
    axes[1].set_title("Estimated unmixing matrix")
    axes[1].set_xlabel("Observation")
    axes[1].set_ylabel("Component")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Generate ICA tutorial figures and print their paths."""

    figures_dir = make_output_dir()
    mixed, sources, mixing = synthesize_sources()
    time = np.linspace(0, 8, mixed.shape[0])

    ica = run_ica(mixed, n_components=3)
    recovered = ica.transform(mixed)

    sources_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_sources_vs_recovered(sources, mixed, recovered, time, sources_path)

    matrices_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_mixing_matrices(mixing, ica.components_, matrices_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()
