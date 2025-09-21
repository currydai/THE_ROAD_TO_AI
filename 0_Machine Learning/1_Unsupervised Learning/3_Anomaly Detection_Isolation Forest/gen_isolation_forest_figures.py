#!/usr/bin/env python3
"""Generate figures for the Isolation Forest anomaly detection tutorial."""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FIGURE_LIST = [
    "isolation_forest_decision.png",
    "isolation_forest_score_hist.png",
]


def make_output_dir() -> str:
    """Return the path to the figures directory, creating it if needed."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def synthesize_data(seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 2D dataset with a dense cluster and scattered anomalies."""

    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=[0.0, 0.0], scale=[0.8, 0.6], size=(800, 2))
    ring_angles = rng.uniform(0.0, 2 * np.pi, size=200)
    ring_radius = rng.normal(loc=3.5, scale=0.2, size=200)
    ring = np.column_stack((ring_radius * np.cos(ring_angles), ring_radius * np.sin(ring_angles)))
    anomalies = rng.uniform(low=-6.0, high=6.0, size=(60, 2))
    points = np.vstack((normal, ring, anomalies))
    labels = np.hstack((np.ones(len(normal)), np.ones(len(ring)), -np.ones(len(anomalies))))

    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    return points, labels


def plot_decision_map(points: np.ndarray, scores: np.ndarray, predictions: np.ndarray, output_path: str) -> None:
    """Scatter plot of anomaly scores with predicted outliers highlighted."""

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
    ax.set_title("Isolation Forest anomaly score map")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Decision function (higher = more normal)")
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_hist(scores: np.ndarray, predictions: np.ndarray, output_path: str) -> None:
    """Histogram contrasting scores of inliers and outliers."""

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    inlier_scores = scores[predictions == 1]
    outlier_scores = scores[predictions == -1]
    bins = np.linspace(scores.min(), scores.max(), 30)

    ax.hist(inlier_scores, bins=bins, alpha=0.7, label="Inliers", color="#1f77b4", edgecolor="k")
    ax.hist(outlier_scores, bins=bins, alpha=0.7, label="Outliers", color="#d62728", edgecolor="k")
    ax.set_xlabel("Decision function score")
    ax.set_ylabel("Count")
    ax.set_title("Isolation Forest score distribution")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    points, true_labels = synthesize_data()

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination=0.08,
        random_state=42,
    )
    model.fit(points)
    scores = model.decision_function(points)
    predictions = model.predict(points)

    decision_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_decision_map(points, scores, predictions, decision_path)

    hist_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_score_hist(scores, predictions, hist_path)

    for figure_name in FIGURE_LIST:
        print(os.path.join(figures_dir, figure_name))


if __name__ == "__main__":
    main()