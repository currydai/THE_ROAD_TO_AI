#!/usr/bin/env python3
"""Generate figures for the Local Outlier Factor tutorial."""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

FIGURE_LIST = [
    "lof_decision_map.png",
    "lof_score_hist.png",
]


def make_output_dir() -> str:
    """Return the figures directory path, creating it if needed."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """Create clustered data with scattered anomalies for LOF demo."""

    rng = np.random.default_rng(seed)

    cluster1 = rng.normal(loc=[-1.5, 0.0], scale=[0.6, 0.4], size=(400, 2))
    cluster2 = rng.normal(loc=[1.6, 1.2], scale=[0.7, 0.5], size=(380, 2))
    cluster3 = rng.normal(loc=[0.5, -1.8], scale=[0.5, 0.6], size=(320, 2))
    anomalies = rng.uniform(low=-5.5, high=5.5, size=(70, 2))

    points = np.vstack((cluster1, cluster2, cluster3, anomalies))
    labels = np.hstack((np.ones(len(cluster1)), np.ones(len(cluster2)), np.ones(len(cluster3)), -np.ones(len(anomalies))))

    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    return points, labels


def plot_decision_map(points: np.ndarray, scores: np.ndarray, predictions: np.ndarray, output_path: str) -> None:
    """Scatter plot of LOF scores highlighting detected outliers."""

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=scores,
        cmap="coolwarm",
        alpha=0.85,
        s=35,
        edgecolor="k",
        linewidth=0.3,
    )
    ax.scatter(
        points[predictions == -1, 0],
        points[predictions == -1, 1],
        facecolor="none",
        edgecolor="black",
        s=80,
        linewidth=1.0,
        label="Predicted outlier",
    )
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")
    ax.set_title("Local Outlier Factor scores")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("LOF score (higher = more anomalous)")
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_hist(scores: np.ndarray, predictions: np.ndarray, output_path: str) -> None:
    """Histogram comparing LOF scores for inliers vs outliers."""

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    inlier_scores = scores[predictions == 1]
    outlier_scores = scores[predictions == -1]
    bins = np.linspace(scores.min(), scores.max(), 28)

    ax.hist(inlier_scores, bins=bins, alpha=0.7, label="Inliers", color="#1f77b4", edgecolor="k")
    ax.hist(outlier_scores, bins=bins, alpha=0.7, label="Outliers", color="#d62728", edgecolor="k")
    ax.set_xlabel("LOF score")
    ax.set_ylabel("Count")
    ax.set_title("LOF score distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    points, true_labels = synthesize_data()

    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.08)
    predictions = lof.fit_predict(points)
    scores = -lof.negative_outlier_factor_

    decision_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_decision_map(points, scores, predictions, decision_path)

    hist_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_score_hist(scores, predictions, hist_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()