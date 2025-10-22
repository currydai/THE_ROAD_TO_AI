"""
Generate illustrative figures for the multimodality and large models tutorials.

Outputs saved under figures/:
  - clip_multimodal_alignment.png
  - lora_rank_update.png
  - rag_pipeline.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

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


def plot_clip_multimodal_alignment() -> None:
    _configure_matplotlib()
    rng = np.random.default_rng(42)

    def make_cluster(center: Sequence[float], cov: np.ndarray, n: int) -> np.ndarray:
        return rng.multivariate_normal(center, cov, size=n)

    image_clusters = [
        make_cluster((0.5, 0.2), np.array([[0.04, 0.01], [0.01, 0.03]]), 120),
        make_cluster((-0.6, 0.3), np.array([[0.05, -0.02], [-0.02, 0.04]]), 120),
        make_cluster((0.1, -0.7), np.array([[0.03, 0.0], [0.0, 0.05]]), 120),
    ]
    text_clusters = [
        make_cluster((0.55, 0.25), np.array([[0.02, 0.0], [0.0, 0.02]]), 120),
        make_cluster((-0.58, 0.28), np.array([[0.02, 0.0], [0.0, 0.02]]), 120),
        make_cluster((0.12, -0.65), np.array([[0.02, 0.0], [0.0, 0.02]]), 120),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["Images: CLIP", "Images: Flamingo Resampler", "Images: LLaVA"]
    for cluster, color, label in zip(image_clusters, colors, labels):
        ax.scatter(cluster[:, 0], cluster[:, 1], label=label, alpha=0.4, c=color, edgecolors="none")

    text_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    text_labels = ["Text prompts", "Instructions", "Caption tokens"]
    for cluster, color, label in zip(text_clusters, text_colors, text_labels):
        ax.scatter(cluster[:, 0], cluster[:, 1], marker="x", c=color, label=label, alpha=0.9)

    ax.annotate(
        "CLIP aligns image/text embeddings\nvia contrastive loss",
        xy=(0.6, 0.3),
        xytext=(1.2, 0.9),
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    )
    ax.annotate(
        "Flamingo gates cross-modal info\ninto frozen language model",
        xy=(-0.5, 0.35),
        xytext=(-1.5, 0.9),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
    )
    ax.annotate(
        "LLaVA projects CLIP features\ninto Vicuna hidden space",
        xy=(0.1, -0.65),
        xytext=(0.9, -1.0),
        arrowprops=dict(arrowstyle="->", color="#2ca02c"),
    )

    ax.set_xlabel("Embedding dimension 1")
    ax.set_ylabel("Embedding dimension 2")
    ax.set_title("Multimodal Embedding Alignment Across Architectures")
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.4, 1.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clip_multimodal_alignment.png", dpi=300)
    plt.close(fig)


def plot_lora_rank_update() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axis("off")

    ax.text(0.1, 0.83, "Frozen weight $\\mathbf{W}_0$", fontsize=12, fontweight="bold", color="#1f77b4")
    ax.add_patch(patches.Rectangle((0.1, 0.25), 0.25, 0.55, edgecolor="#1f77b4", facecolor=(0.92, 0.95, 1.0)))

    ax.text(0.43, 0.83, "LoRA down $\\mathbf{A}$", fontsize=12, fontweight="bold", color="#ff7f0e")
    ax.add_patch(patches.Rectangle((0.43, 0.55), 0.18, 0.25, edgecolor="#ff7f0e", facecolor=(1.0, 0.95, 0.9)))

    ax.text(0.68, 0.83, "LoRA up $\\mathbf{B}$", fontsize=12, fontweight="bold", color="#2ca02c")
    ax.add_patch(patches.Rectangle((0.68, 0.55), 0.18, 0.25, edgecolor="#2ca02c", facecolor=(0.9, 1.0, 0.92)))

    ax.add_patch(
        patches.FancyArrow(0.6, 0.4, 0.0, -0.12, width=0.015, length_includes_head=True, color="#555555")
    )
    ax.text(0.62, 0.33, "$\\frac{\\alpha}{r}\\mathbf{B}(\\mathbf{A}h)$", fontsize=11, color="#2ca02c")

    ax.add_patch(
        patches.FancyArrow(0.22, 0.4, 0.0, -0.12, width=0.015, length_includes_head=True, color="#555555")
    )
    ax.text(0.24, 0.33, "$\\mathbf{W}_0 h$", fontsize=11, color="#1f77b4")

    ax.add_patch(
        patches.FancyArrow(0.36, 0.22, 0.28, 0.0, width=0.015, length_includes_head=True, color="#9467bd")
    )
    ax.text(0.5, 0.16, "Summation", fontsize=10, color="#9467bd")

    ax.add_patch(patches.Rectangle((0.36, 0.08), 0.28, 0.1, edgecolor="#9467bd", facecolor=(0.96, 0.92, 1.0)))
    ax.text(0.5, 0.13, "Output $y$", ha="center", va="center", fontsize=11)

    ax.text(
        0.1,
        0.04,
        "Only $\\mathbf{A}, \\mathbf{B}$ updated (rank $r$). Original weights remain frozen.\nScaling $\\alpha/r$ controls adaptation strength.",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "lora_rank_update.png", dpi=300)
    plt.close(fig)


def plot_rag_pipeline() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    boxes = [
        (0.2, 1.6, "User Query\n\"Summarize quantum paper\"", "#1f77b4"),
        (2.0, 1.6, "Dense Retriever\n(DPR / Contriever)", "#ff7f0e"),
        (3.8, 2.2, "Document Store\n(Vector index + metadata)", "#2ca02c"),
        (3.8, 1.0, "Cache\n(Recent hits)", "#2ca02c"),
        (5.6, 1.6, "Fusion-in-Decoder\nContext Builder", "#9467bd"),
        (7.4, 1.6, "Generator\n(LLM + LoRA)", "#d62728"),
        (9.0, 1.6, "Response\n+ sources", "#17becf"),
    ]

    for x, y, text, color in boxes:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                1.4,
                0.9,
                boxstyle="round,pad=0.2",
                edgecolor=color,
                facecolor=(0.95, 0.97, 1.0),
                linewidth=1.5,
            )
        )
        ax.text(x + 0.7, y + 0.45, text, ha="center", va="center", fontsize=10, color=color)

    def arrow(x0: float, y0: float, x1: float, y1: float, text: str | None = None) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#555555"),
        )
        if text:
            ax.text((x0 + x1) / 2, y0 + 0.35, text, ha="center", va="center", fontsize=9)

    arrow(1.6, 2.05, 2.0, 2.05, "Encode query")
    arrow(3.4, 2.05, 3.8, 2.05, "Top-k retrieval")
    arrow(3.4, 2.05, 3.8, 1.35, "Cache hit")
    arrow(5.2, 2.05, 5.6, 2.05, "Concatenate passages")
    arrow(7.0, 2.05, 7.4, 2.05, "Context tokens")
    arrow(8.8, 2.05, 9.0, 2.05, None)

    ax.text(
        5.0,
        0.6,
        "Monitoring: latency, hit-rate, hallucination checks\nSafety: filtered retrieval, policy guardrails",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#9467bd"),
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "rag_pipeline.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_clip_multimodal_alignment()
    plot_lora_rank_update()
    plot_rag_pipeline()


if __name__ == "__main__":
    main()
