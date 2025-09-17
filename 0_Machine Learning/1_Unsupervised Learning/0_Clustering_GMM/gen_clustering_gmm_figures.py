#!/usr/bin/env python3
"""Generate figures for the Gaussian Mixture Model tutorial."""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

FIGURE_LIST: List[str] = [
    "gmm_density_contours.png",
    "gmm_bic_curve.png",
]


def make_output_dir() -> str:
    """Return figures directory path, creating it if necessary."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 5) -> np.ndarray:
    """Generate synthetic 2D data from three Gaussian blobs with noise."""

    rng = np.random.RandomState(seed)

    blob_a = rng.multivariate_normal(mean=(-2.0, 1.5), cov=[[0.35, 0.1], [0.1, 0.25]], size=180)
    blob_b = rng.multivariate_normal(mean=(1.8, 1.7), cov=[[0.3, -0.12], [-0.12, 0.4]], size=200)
    blob_c = rng.multivariate_normal(mean=(0.0, -2.2), cov=[[0.4, 0.05], [0.05, 0.2]], size=190)
    noise = rng.normal(loc=0.0, scale=1.1, size=(80, 2))

    points = np.vstack((blob_a, blob_b, blob_c, noise))
    points -= points.mean(axis=0)
    points /= points.std(axis=0)
    return points


def plot_density_contours(points: np.ndarray, model: GaussianMixture, output_path: str) -> None:
    """Plot data points and GMM density contours with covariance ellipses."""

    x_min, y_min = points.min(axis=0) - 0.5
    x_max, y_max = points.max(axis=0) + 0.5
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    densities = np.exp(model.score_samples(grid_points)).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    scatter = ax.scatter(points[:, 0], points[:, 1], s=20, alpha=0.45, c="#555555", edgecolor="k", linewidth=0.2)
    contour_levels = np.linspace(densities.min(), densities.max(), 12)
    ax.contour(grid_x, grid_y, densities, levels=contour_levels[2:], cmap="viridis", alpha=0.85)

    for mean, covar, weight in zip(model.means_, model.covariances_, model.weights_):
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
        )
        ax.add_patch(ellipse)

    ax.set_title("GMM Density Contours (K=3)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bic_curve(points: np.ndarray, output_path: str, k_max: int = 6) -> None:
    """Plot BIC scores for a range of component counts."""

    ks = list(range(1, k_max + 1))
    bic_scores = []

    for k in ks:
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            init_params="kmeans",
            max_iter=400,
            reg_covar=1e-6,
            random_state=42,
        )
        model.fit(points)
        bic_scores.append(model.bic(points))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(ks, bic_scores, marker="o", linewidth=1.6, color="#1f77b4")
    ax.set_title("BIC Across Component Counts")
    ax.set_xlabel("Number of components (K)")
    ax.set_ylabel("BIC")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Generate tutorial figures and print their absolute paths."""

    figures_dir = make_output_dir()
    points = synthesize_data()

    gmm_model = GaussianMixture(
        n_components=3,
        covariance_type="full",
        init_params="kmeans",
        max_iter=400,
        reg_covar=1e-6,
        random_state=42,
    )
    gmm_model.fit(points)

    density_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_density_contours(points, gmm_model, density_path)

    bic_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_bic_curve(points, bic_path, k_max=6)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()
