"""
Generate figures for the model compression and deployment tutorials.

Outputs (PNG) saved under figures/:
  - compression_overview.png
  - deployment_toolchain.png
  - latency_breakdown.png
  - accelerated_inference_strategies.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def plot_compression_overview() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    stages = [
        ("Original Model", "baseline\nparameters: 120M\nsize: 480 MB"),
        ("Pruned Model", "structured pruning\nparameters: 60M\nsize: 240 MB"),
        ("Distilled Student", "teacher-student\nparameters: 30M\nsize: 120 MB"),
        ("Quantized Model", "INT8 runtime\nparameters: 30M\nsize: 30 MB"),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    x = 0.5
    for (title, desc), color in zip(stages, colors):
        box = patches.FancyBboxPatch(
            (x, 1.2),
            1.6,
            1.2,
            boxstyle="round,pad=0.2",
            linewidth=1.5,
            edgecolor=color,
            facecolor=(0.95, 0.97, 1.0),
        )
        ax.add_patch(box)
        ax.text(x + 0.8, 2.3, title, ha="center", va="center", fontsize=12, color=color, fontweight="bold")
        ax.text(x + 0.8, 1.8, desc, ha="center", va="center", fontsize=10)
        x += 2.1

    arrow_positions = [1.9, 4.0, 6.1]
    for xpos in arrow_positions:
        ax.annotate(
            "",
            xy=(xpos + 0.3, 1.8),
            xytext=(xpos, 1.8),
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#555555"),
        )

    ax.text(
        2.5,
        0.6,
        "\n".join(
            [
                "Top-1 accuracy (%)",
                "baseline: 78.8",
                "pruned: 78.1",
                "distilled: 78.5",
                "quantized: 77.9",
            ]
        ),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="#1f77b4"),
        fontsize=10,
    )

    ax.text(
        5.9,
        0.6,
        "\n".join(
            [
                "Latency on ARM CPU (ms)",
                "baseline: 120",
                "pruned: 95",
                "distilled: 70",
                "quantized: 42",
            ]
        ),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="#2ca02c"),
        fontsize=10,
    )

    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 3.5)
    ax.set_title("Compression Workflow Overview")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "compression_overview.png", dpi=300)
    plt.close(fig)


def plot_deployment_toolchain() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    columns = [
        ("Training Frameworks", ["PyTorch", "TensorFlow", "JAX"]),
        ("Export / IR", ["ONNX", "TF SavedModel", "TorchScript"]),
        ("Optimizers", ["TensorRT", "ONNX Runtime", "TFLite Converter"]),
        ("Targets", ["Server GPU", "Edge GPU", "Mobile CPU", "NPU/TPU"]),
    ]

    x_positions = [0.5, 3.0, 5.5, 8.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for (title, entries), x, color in zip(columns, x_positions, colors):
        rect = patches.FancyBboxPatch(
            (x, 0.8),
            1.8,
            3.0,
            boxstyle="round,pad=0.2",
            linewidth=1.5,
            edgecolor=color,
            facecolor=(0.94, 0.96, 1.0),
        )
        ax.add_patch(rect)
        ax.text(x + 0.9, 3.5, title, ha="center", va="center", fontsize=12, color=color, fontweight="bold")
        y = 3.0
        for entry in entries:
            ax.text(x + 0.9, y, entry, ha="center", va="center", fontsize=10)
            y -= 0.6

    for x_start, x_end in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate(
            "",
            xy=(x_end, 2.4),
            xytext=(x_start + 1.8, 2.4),
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#555555"),
        )

    ax.text(
        5.0,
        0.4,
        "Delegates / plugins:\nNNAPI, Core ML, CUDA plugins, custom ops",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ff7f0e"),
    )

    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 4.2)
    ax.set_title("Deployment Toolchain Across Frameworks and Targets")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deployment_toolchain.png", dpi=300)
    plt.close(fig)


def plot_latency_breakdown() -> None:
    _configure_matplotlib()
    components = ["Compute", "Memory", "Framework", "I/O", "Scheduling"]
    baseline = np.array([65, 25, 10, 8, 6])
    optimized = np.array([38, 12, 6, 5, 4])

    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline, width, label="Baseline", color="#1f77b4")
    ax.bar(x + width / 2, optimized, width, label="Optimized", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Breakdown Before and After Optimization")
    ax.legend()

    for idx, (b, o) in enumerate(zip(baseline, optimized)):
        ax.text(idx - width / 2, b + 1, f"{b:.0f}", ha="center", fontsize=9)
        ax.text(idx + width / 2, o + 1, f"{o:.0f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "latency_breakdown.png", dpi=300)
    plt.close(fig)


def plot_accelerated_inference_strategies() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    layers = [
        ("Model Level", ["Pruning", "Distillation", "Low-rank factorization", "Mixture-of-experts routing"]),
        ("Kernel Level", ["Operator fusion", "Winograd/FFT conv", "Sparse kernels", "Auto-tuned schedules"]),
        ("Serving Level", ["Dynamic batching", "Speculative decoding", "Request coalescing", "Multi-model endpoints"]),
        ("Hardware Level", ["INT8 Tensor Cores", "Edge TPU", "FPGA overlays", "CPU vectorization"]),
    ]

    y_positions = [2.9, 2.1, 1.3, 0.5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for (title, bullets), y, color in zip(layers, y_positions, colors):
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.5, y),
                8.5,
                0.7,
                boxstyle="round,pad=0.2",
                linewidth=1.2,
                edgecolor=color,
                facecolor=(0.95, 0.97, 1.0),
            )
        )
        ax.text(0.8, y + 0.35, title, ha="left", va="center", fontsize=12, color=color, fontweight="bold")
        ax.text(
            3.0,
            y + 0.35,
            " | ".join(bullets),
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 3.7)
    ax.set_title("Inference Acceleration Strategies Across the Stack")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accelerated_inference_strategies.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_compression_overview()
    plot_deployment_toolchain()
    plot_latency_breakdown()
    plot_accelerated_inference_strategies()


if __name__ == "__main__":
    main()
