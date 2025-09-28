#!/usr/bin/env python3
"""Generate figures for the core CNN tutorial."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def figures_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_structure_receptive_field(out_dir: Path) -> None:
    """CNN receptive field structure illustration"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_aspect("equal")

    def draw_feature_map(origin_x, origin_y, size, color, label, step=1.0):
        for i in range(size):
            for j in range(size):
                rect = patches.Rectangle(
                    (origin_x + j * step, origin_y + (size - i - 1) * step),
                    step,
                    step,
                    linewidth=1.0,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.15,   # 半透明填充
                )
                ax.add_patch(rect)
        ax.text(
            origin_x + size * step / 2,
            origin_y + size * step + 0.4,
            label,
            ha="center",
            fontsize=11,
        )

    # 各层 feature map
    draw_feature_map(0, 0, 5, "#1f77b4", "Input 5x5x3")
    draw_feature_map(6, 0.5, 4, "#ff7f0e", "Conv1 3x3 stride 1")
    draw_feature_map(11, 1, 3, "#2ca02c", "MaxPool 2x2 stride 2")
    draw_feature_map(15, 1.5, 3, "#d62728", "Conv2 dilation 2")

    # 箭头
    arrows = [
        ((4.5, 2.5), (5.8, 2.5)),
        ((9.5, 2.5), (10.8, 2.5)),
        ((13.8, 2.5), (15.0, 2.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.5))

    # 说明文字
    ax.text(7.5, -1.1, "padding=1", fontsize=9)
    ax.text(12.5, -1.1, "stride=2", fontsize=9)
    ax.text(16.5, -1.1, "dilation=2", fontsize=9)

    # ✅ 固定坐标范围，避免内容缩角落
    ax.set_xlim(-1, 20)
    ax.set_ylim(-2, 8)

    fig.savefig(out_dir / "cnn_structure_receptive_field.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_evolution(out_dir: Path) -> None:
    """CNN depth & parameter evolution"""
    models = [
        (1998, "LeNet", 7),
        (2012, "AlexNet", 8),
        (2014, "VGG", 16),
        (2015, "ResNet", 152),
        (2015, "Inception", 48),
        (2019, "EfficientNet", 82),
    ]
    years = np.array([m[0] for m in models])
    depths = np.array([m[2] for m in models])

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(years, depths, marker="o", color="#1f77b4", linewidth=2, label="Depth")
    ax1.set_ylabel("Depth (layers)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    params = np.array([0.06, 61, 138, 25, 24, 19])
    ax2.plot(years, params, marker="s", color="#d62728", linewidth=2, label="Parameters (M)")
    ax2.set_ylabel("Parameters (millions)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    for x, depth, param, name in zip(years, depths, params, [m[1] for m in models]):
        ax1.text(x, depth + 5, name, ha="center", fontsize=9)
        ax2.text(x, param + 6, f"{param:.0f}", ha="center", fontsize=8, color="#d62728")

    ax1.set_xlabel("Year")
    ax1.set_title("Evolution of CNN depth and parameter counts")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(1996, 2021)

    # ✅ y 轴范围固定，避免文字挤掉
    ax1.set_ylim(0, 170)
    ax2.set_ylim(0, 160)

    fig.savefig(out_dir / "cnn_architecture_evolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_applications_overview(out_dir: Path) -> None:
    """Overview of CNN applications"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")

    sections = [
        ("Image Classification", ["Backbone", "Global Average Pool", "Softmax"]),
        ("Object Detection", ["Backbone", "Feature Pyramid", "Bounding Box Head", "NMS"]),
        ("Semantic Segmentation", ["Encoder", "Atrous Conv", "Decoder", "Pixel-wise Argmax"]),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, ((title, stages), color) in enumerate(zip(sections, colors)):
        x0 = idx * 3.1
        box = patches.FancyBboxPatch(
            (x0, 0.4), 2.9, 3.2,
            boxstyle="round,pad=0.3", linewidth=1.6,
            edgecolor=color, facecolor=color, alpha=0.15
        )
        ax.add_patch(box)
        ax.text(x0 + 1.45, 3.2, title, ha="center", fontsize=12, fontweight="bold", color="#333333")
        for stage_idx, stage in enumerate(stages):
            ax.text(x0 + 1.45, 2.5 - stage_idx * 0.6, stage, ha="center", fontsize=10, color="#333333")

    ax.text(4.5, 4.0, "CNN application pipelines", ha="center", fontsize=13)

    # ✅ 固定显示范围
    ax.set_xlim(-1, 10)
    ax.set_ylim(0, 5)

    fig.savefig(out_dir / "cnn_applications_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = figures_dir()
    plot_structure_receptive_field(out_dir)
    plot_architecture_evolution(out_dir)
    plot_applications_overview(out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
