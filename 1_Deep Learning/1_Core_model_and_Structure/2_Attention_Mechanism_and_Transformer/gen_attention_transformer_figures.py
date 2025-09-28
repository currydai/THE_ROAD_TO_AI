#!/usr/bin/env python3
"""Generate improved figures for attention and transformer tutorials."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import to_rgba


def figures_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_attention_scores(out_dir: Path) -> None:
    seq_len = 6
    scores = np.random.uniform(0.1, 1.0, size=(seq_len, seq_len))
    scores = scores / scores.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(scores, cmap="viridis")

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels([f"k{j}" for j in range(seq_len)], fontsize=10)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels([f"q{i}" for i in range(seq_len)], fontsize=10)

    for i in range(seq_len):
        for j in range(seq_len):
            ax.text(j, i, f"{scores[i, j]:.2f}",
                    ha="center", va="center",
                    color="white", fontsize=8, weight="bold")

    ax.set_xlabel("Keys", fontsize=11)
    ax.set_ylabel("Queries", fontsize=11)
    ax.set_title("Scaled Dot-Product Attention Weights", fontsize=13, weight="bold")

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(out_dir / "attention_scores_heatmap.png", dpi=300)
    plt.close(fig)


def plot_multihead_attention(out_dir: Path) -> None:
    heads = 4
    fig, axes = plt.subplots(1, heads, figsize=(12, 3.5))
    for idx, ax in enumerate(axes):
        scores = np.random.uniform(0.1, 1.0, size=(6, 6))
        scores = scores / scores.sum(axis=1, keepdims=True)
        ax.imshow(scores, cmap="plasma")
        ax.set_title(f"Head {idx+1}", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Multi-Head Self-Attention Patterns", fontsize=14, weight="bold")
    fig.subplots_adjust(wspace=0.25, top=0.82)
    fig.savefig(out_dir / "self_attention_heads.png", dpi=300)
    plt.close(fig)


def plot_transformer_stack(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    def draw_block(x: float, y: float, text: str, color: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y), 2.4, 0.9,
            boxstyle="round,pad=0.25,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=color,
            facecolor=to_rgba(color, 0.25)
        )
        ax.add_patch(rect)
        ax.text(x + 1.2, y + 0.45, text,
                ha="center", va="center",
                fontsize=11, weight="bold", color="#111111")

    # --- Encoder stack ---
    encoder_layers = 4
    y_start = 0.6
    for i in range(encoder_layers):
        y0 = y_start + i * 2.2
        draw_block(0.5, y0, "Multi-Head Attention", "#1f77b4")
        draw_block(0.5, y0 + 1.0, "Feed-Forward", "#ff7f0e")
        if i < encoder_layers - 1:
            ax.annotate("",
                        xy=(1.7, y0 + 2.0),
                        xytext=(1.7, y0 + 1.2),
                        arrowprops=dict(arrowstyle="->",
                                        linewidth=1.6,
                                        color="#444444"))

    # --- Decoder layer ---
    dx = 5.0
    draw_block(dx, 6.0, "Masked MHA", "#1f77b4")
    draw_block(dx, 7.2, "Cross Attention", "#2ca02c")
    draw_block(dx, 8.4, "Feed-Forward", "#ff7f0e")
    ax.annotate("",
                xy=(dx + 1.2, 5.8),
                xytext=(dx + 1.2, 4.5),
                arrowprops=dict(arrowstyle="->",
                                linewidth=1.6,
                                color="#444444"))

    # Titles
    ax.text(1.7, 10.0, "Encoder Stack",
            ha="center", fontsize=16, weight="bold")
    ax.text(dx + 1.2, 10.3, "Decoder Layer",
            ha="center", fontsize=16, weight="bold")

    # 范围控制
    ax.set_xlim(-0.5, 9.0)
    ax.set_ylim(0.0, 11.0)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    fig.savefig(out_dir / "transformer_layer_stack.png", dpi=300)
    plt.close(fig)

def plot_applications(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    sections = [
        ("NLP", ["Machine Translation", "Summarization", "QA / Dialogue"]),
        ("Vision & Cross-modal", ["Vision Transformer", "Text-Image Alignment (CLIP)", "Video Transformers"]),
        ("Multimodal LLM", ["Instruction Tuning", "Retrieval Augmentation", "Tool Use"]),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, ((title, bullets), color) in enumerate(zip(sections, colors)):
        x0 = idx * 3.3
        box = patches.FancyBboxPatch(
            (x0, 0.6), 3.0, 3.0,
            boxstyle="round,pad=0.2,rounding_size=0.05",
            linewidth=1.3,
            edgecolor=color,
            facecolor=to_rgba(color, 0.15)
        )
        ax.add_patch(box)
        ax.text(x0 + 1.5, 3.0, title, ha="center",
                fontsize=13, weight="bold", color="#222222")
        for j, bullet in enumerate(bullets):
            ax.text(x0 + 1.5, 2.3 - j * 0.7, bullet,
                    ha="center", fontsize=10, color="#333333")

    ax.text(5.0, 4.2, "Attention & Transformer Applications",
            ha="center", fontsize=17, weight="bold")

    ax.set_xlim(-0.2, 10.0)
    ax.set_ylim(0.0, 5.0)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)
    fig.savefig(out_dir / "attention_transformer_applications.png", dpi=300)
    plt.close(fig)


def main() -> None:
    out_dir = figures_dir()
    plot_attention_scores(out_dir)
    plot_multihead_attention(out_dir)
    plot_transformer_stack(out_dir)
    plot_applications(out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
