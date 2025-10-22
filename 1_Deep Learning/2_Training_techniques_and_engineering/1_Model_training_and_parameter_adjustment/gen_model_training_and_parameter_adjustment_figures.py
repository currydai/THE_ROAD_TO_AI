"""
Generate illustrative figures for the model training and parameter adjustment tutorials.

Outputs (PNG) in the local figures/ directory:
  - hyperparameter_landscape.png
  - early_stopping_checkpoint.png
  - distributed_training_topologies.png
  - scaling_efficiency.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import patches
from matplotlib import cm

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def plot_hyperparameter_landscape() -> None:
    _configure_matplotlib()
    lr = np.logspace(-5, -1, 50)
    batch = np.linspace(32, 1024, 50)
    optimizers = ["SGD", "AdamW", "Lion"]

    LR, B = np.meshgrid(lr, batch)
    surfaces = {}

    surfaces["SGD"] = 0.75 * np.exp(-((np.log10(LR) + 2.5) ** 2) / 0.7) * np.exp(-((np.log2(B) - 7) ** 2) / 3)
    surfaces["AdamW"] = 0.82 * np.exp(-((np.log10(LR) + 2.8) ** 2) / 0.6) * np.exp(-((np.log2(B) - 8) ** 2) / 4) + 0.05
    surfaces["Lion"] = 0.80 * np.exp(-((np.log10(LR) + 2.9) ** 2) / 0.55) * np.exp(-((np.log2(B) - 8.5) ** 2) / 4.5) + 0.03

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"SGD": cm.Blues, "AdamW": cm.Oranges, "Lion": cm.Greens}
    for name in optimizers:
        surf = ax.plot_surface(
            np.log10(LR),
            np.log2(B),
            surfaces[name],
            cmap=colors[name],
            alpha=0.65,
            linewidth=0,
            antialiased=True,
        )
        surf.set_edgecolor("none")

    ax.set_title("Validation Accuracy Landscape")
    ax.set_xlabel("$\\log_{10}(\\eta)$")
    ax.set_ylabel("$\\log_{2}(B)$")
    ax.set_zlabel("Validation accuracy")
    ax.view_init(elev=28, azim=-135)
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda _, pos: f"{_:.2f}"))

    proxy = [patches.Patch(color=cm.Blues(0.7), label="SGD"),
             patches.Patch(color=cm.Oranges(0.7), label="AdamW"),
             patches.Patch(color=cm.Greens(0.7), label="Lion")]
    ax.legend(handles=proxy, loc="upper left", bbox_to_anchor=(0.05, 0.95))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "hyperparameter_landscape.png", dpi=300)
    plt.close(fig)


def plot_early_stopping_checkpoint() -> None:
    _configure_matplotlib()
    steps = np.arange(0, 160)
    metric = 0.7 + 0.15 * (1 - np.exp(-steps / 40)) - 0.05 * np.maximum(0, np.sin(steps / 10)) + 0.002 * np.random.RandomState(42).randn(len(steps))
    metric = np.clip(metric, 0.68, None)
    patience = 15
    best = np.maximum.accumulate(metric)
    trigger = np.full_like(metric, np.nan)

    for t in range(len(steps)):
        start = max(0, t - patience)
        if best[t] <= best[start] + 1e-3 and t - start >= patience - 1:
            trigger[t] = metric[t]
            break

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, metric, label="Validation metric", color="#1f77b4")
    ax.plot(steps, best, label="Best so far", linestyle="--", color="#ff7f0e")

    checkpoint_steps = [40, 80, 120]
    checkpoint_metrics = metric[checkpoint_steps]
    ax.scatter(checkpoint_steps, checkpoint_metrics, marker="D", color="#2ca02c", label="Checkpoint", zorder=5)
    if np.isfinite(trigger).any():
        t_stop = steps[np.isfinite(trigger)][0]
        ax.axvline(t_stop, color="#d62728", linestyle=":", label="Early stop")
        ax.annotate(
            f"Early stop @ step {t_stop}",
            xy=(t_stop, metric[t_stop]),
            xytext=(t_stop + 10, metric[t_stop] + 0.02),
            arrowprops=dict(arrowstyle="->", color="#d62728"),
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Early Stopping with Rolling Checkpoints")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "early_stopping_checkpoint.png", dpi=300)
    plt.close(fig)


def plot_distributed_training_topologies() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    def add_cluster(x: float, y: float, width: float, height: float, title: str, nodes: List[str], color: str) -> None:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.2",
                linewidth=1.5,
                edgecolor=color,
                facecolor=(1, 1, 1),
            )
        )
        ax.text(x + width / 2, y + height - 0.25, title, ha="center", va="top", fontsize=12, color=color)
        spacing = height / (len(nodes) + 1)
        for idx, node in enumerate(nodes, start=1):
            ax.text(x + width / 2, y + height - spacing * idx - 0.1, node, ha="center", va="center", fontsize=10)

    add_cluster(0.2, 1.0, 2.5, 1.5, "Data Parallel", ["GPU 0\nBatch 0", "GPU 1\nBatch 1", "GPU 2\nBatch 2", "GPU 3\nBatch 3"], "#1f77b4")
    ax.annotate(
        "",
        xy=(1.45, 0.9),
        xytext=(1.45, 0.7),
        arrowprops=dict(arrowstyle="|-|", color="#1f77b4"),
    )
    ax.text(1.45, 0.62, "All-reduce", ha="center", va="center", fontsize=9)

    add_cluster(3.3, 1.0, 2.5, 1.5, "Pipeline Parallel", ["Stage 0\nLayers 0-11", "Stage 1\nLayers 12-23", "Stage 2\nLayers 24-35", "Stage 3\nLayers 36-47"], "#ff7f0e")
    ax.annotate(
        "",
        xy=(3.35, 0.75),
        xytext=(5.55, 0.75),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
    )
    ax.text(4.45, 0.62, "Activations forward", ha="center", va="center", fontsize=9)
    ax.annotate(
        "",
        xy=(5.55, 0.55),
        xytext=(3.35, 0.55),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
    )
    ax.text(4.45, 0.42, "Gradients backward", ha="center", va="center", fontsize=9)

    add_cluster(6.4, 1.0, 2.5, 1.5, "Hybrid", ["DP Group 0\nStages 0-1", "DP Group 1\nStages 2-3", "Tensor Shard A", "Tensor Shard B"], "#2ca02c")
    ax.annotate(
        "",
        xy=(6.5, 0.75),
        xytext=(8.7, 0.75),
        arrowprops=dict(arrowstyle="<|-|>", color="#2ca02c"),
    )
    ax.text(7.6, 0.62, "Tensor all-reduce", ha="center", va="center", fontsize=9)

    ax.text(4.8, 2.7, "Common Distributed Training Topologies", ha="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distributed_training_topologies.png", dpi=300)
    plt.close(fig)


def plot_scaling_efficiency() -> None:
    _configure_matplotlib()
    workers = np.array([1, 2, 4, 8, 16, 32, 64])
    weak_scaling = 0.92 / (1 + 0.03 * np.log2(workers)) + 0.05
    strong_scaling = 1 / (1 + 0.1 * (workers - 1)) * 0.95

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(workers, weak_scaling, marker="o", label="Weak scaling")
    ax.plot(workers, strong_scaling, marker="s", label="Strong scaling")
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Parallel efficiency")
    ax.set_title("Scaling Efficiency vs. Cluster Size")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_efficiency.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_hyperparameter_landscape()
    plot_early_stopping_checkpoint()
    plot_distributed_training_topologies()
    plot_scaling_efficiency()


if __name__ == "__main__":
    main()
