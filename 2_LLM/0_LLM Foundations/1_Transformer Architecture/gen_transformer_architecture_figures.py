"""
Generate illustrative figures for transformer architecture tutorials (non-overlapping, PNG only).

Outputs saved to ./figures:
  - self_attention_flow.png
  - transformer_stack_variants.png
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ======= Global style =======
DPI = 300
FIGSIZE = (12, 6.75)  # 16:9, safe for slides
FS_TITLE = 13
FS_TEXT = 10
FS_TINY = 9

def _configure():
    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "savefig.dpi": DPI,
        "font.size": FS_TEXT,
        "axes.titlesize": FS_TITLE,
        "axes.grid": False,
        "figure.constrained_layout.use": True,
        "savefig.facecolor": "white",
    })

# --------------------------------------------------------------
# 1. Self-Attention Flow
# --------------------------------------------------------------
def plot_self_attention_flow():
    _configure()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    labels = [
        "Input\nEmbeddings",
        "Linear\nProjections",
        "Scaled Dot-Product\nAttention (per head)",
        "Concat\nHeads",
        "Output\nProjection",
    ]

    # box positions (均匀分布)
    x_positions = [0.7, 3.3, 5.9, 8.5, 11.1]
    y = 2.4
    box_w, box_h = 1.9, 1.4

    for x, label, color in zip(x_positions, labels, colors):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.25",
                edgecolor=color,
                facecolor=(0.95, 0.97, 1.0),
                linewidth=1.3,
            )
        )
        ax.text(x + box_w/2, y + box_h/2, label,
                ha="center", va="center", fontsize=FS_TEXT, color=color)

    # arrows
    for i in range(len(x_positions) - 1):
        x0 = x_positions[i] + box_w
        x1 = x_positions[i + 1]
        ax.annotate(
            "",
            xy=(x1, y + box_h/2),
            xytext=(x0, y + box_h/2),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )

    # left tokens
    tokens = ["Token 1", "Token 2", "Token 3", "Token 4"]
    for i, token in enumerate(tokens):
        ax.text(0.9, 0.9 - i * 0.25, token, fontsize=FS_TINY, ha="left")

    # bottom note
    ax.text(
        6.5, 0.25,
        "Each head computes attention in parallel.\nOutputs are concatenated and mixed by $W^O$.",
        ha="center", va="center", fontsize=FS_TEXT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#9467bd")
    )

    ax.set_title("Self-Attention Flow (Per Layer)", pad=8)
    fig.savefig(FIG_DIR / "self_attention_flow.png",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# --------------------------------------------------------------
# 2. Transformer Stack Variants
# --------------------------------------------------------------
def plot_transformer_stack_variants():
    _configure()
    fig, ax = plt.subplots(constrained_layout=True)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    columns = [
        ("Encoder-only", ["Self-Attention", "FFN", "LayerNorm", "Residual"], "#1f77b4"),
        ("Decoder-only", ["Masked Self-Attention", "FFN", "LayerNorm", "Residual"], "#ff7f0e"),
        ("Encoder-Decoder", ["Encoder stack", "Cross-Attention", "Decoder stack"], "#2ca02c"),
        ("Hybrid / RAG", ["Encoder retrieval", "Decoder generation", "Cross modules"], "#9467bd"),
    ]

    box_w, box_h = 2.4, 3.2
    x_positions = [0.9, 4.1, 7.3, 10.5]
    title_y = 5.0
    start_y = 4.3
    line_gap = 0.65

    for (title, entries, color), x in zip(columns, x_positions):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 1.0), box_w, box_h,
                boxstyle="round,pad=0.3",
                edgecolor=color,
                facecolor=(0.95, 0.97, 1.0),
                linewidth=1.3,
            )
        )
        ax.text(x + box_w / 2, title_y, title,
                ha="center", va="center", fontsize=FS_TITLE, color=color, fontweight="bold")

        y = start_y
        for entry in entries:
            ax.text(x + 0.3, y, f"• {entry}", ha="left", va="center", fontsize=FS_TEXT)
            y -= line_gap

    # bottom description
    ax.text(
        7.2, 0.5,
        "Different transformer stacks target varying workloads:\n"
        "Encoders for understanding, decoders for generation,\n"
        "encoder-decoder for conditional tasks, hybrids for retrieval augmentation.",
        ha="center", va="center", fontsize=FS_TEXT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#d62728"),
    )

    ax.set_title("Transformer Stack Variants", pad=8)
    fig.savefig(FIG_DIR / "transformer_stack_variants.png",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# --------------------------------------------------------------
def main():
    plot_self_attention_flow()
    plot_transformer_stack_variants()

if __name__ == "__main__":
    main()
