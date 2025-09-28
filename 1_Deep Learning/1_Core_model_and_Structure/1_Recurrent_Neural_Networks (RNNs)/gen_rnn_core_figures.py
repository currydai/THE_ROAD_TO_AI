#!/usr/bin/env python3
"""Generate illustrative figures for the RNN tutorial (optimized layout)."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def figures_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_rnn_unrolled(out_dir: Path) -> None:
    """Unrolled RNN dynamics"""
    time_steps = 5
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    for t in range(time_steps):
        x = t * 1.8
        # Input
        input_rect = patches.FancyBboxPatch(
            (x, 1.6), 1.0, 0.6, boxstyle="round,pad=0.2",
            linewidth=1.2, facecolor="#1f77b4", edgecolor="#1f77b4"
        )
        hidden_rect = patches.FancyBboxPatch(
            (x, 0.7), 1.0, 0.6, boxstyle="round,pad=0.2",
            linewidth=1.2, facecolor="#ff7f0e", edgecolor="#ff7f0e"
        )
        output_rect = patches.FancyBboxPatch(
            (x, -0.2), 1.0, 0.6, boxstyle="round,pad=0.2",
            linewidth=1.2, facecolor="#2ca02c", edgecolor="#2ca02c"
        )
        ax.add_patch(input_rect)
        ax.add_patch(hidden_rect)
        ax.add_patch(output_rect)

        # Labels
        ax.text(x + 0.5, 1.9, f"$x_{t}$", ha="center", va="center", color="white", fontsize=11)
        ax.text(x + 0.5, 1.0, f"$h_{t}$", ha="center", va="center", color="white", fontsize=11)
        ax.text(x + 0.5, 0.1, f"$y_{t}$", ha="center", va="center", color="white", fontsize=11)

        # Arrows
        if t > 0:
            ax.annotate("", xy=(x, 1.0), xytext=(x - 0.8, 1.0),
                        arrowprops=dict(arrowstyle="->", linewidth=1.5))
        ax.annotate("", xy=(x + 0.5, 1.6), xytext=(x + 0.5, 1.3),
                    arrowprops=dict(arrowstyle="->", linewidth=1.2))
        ax.annotate("", xy=(x + 0.5, 0.7), xytext=(x + 0.5, 0.4),
                    arrowprops=dict(arrowstyle="->", linewidth=1.2))

    # 循环箭头 h_t → h_{t+1}
    ax.annotate("", xy=(8.5, 1.0), xytext=(9.2, 1.0),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, linestyle="--", color="gray"))

    ax.text(4.5, 2.7, "Unrolled RNN over time", ha="center", fontsize=15, fontweight="bold")
    ax.set_xlim(-1, 10.5)
    ax.set_ylim(-1, 3.2)

    fig.savefig(out_dir / "rnn_unrolled_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lstm_gru_comparison(out_dir: Path) -> None:
    """LSTM vs GRU gates with improved layout"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    def draw_cell(origin_x: float, title: str, gates: list[str], colors: list[str],
                  show_c: bool=False) -> None:
        width, height = 4.0, 3.6
        box = patches.FancyBboxPatch(
            (origin_x, 1.0), width, height,
            boxstyle="round,pad=0.25", linewidth=1.5,
            facecolor="#fdfdfd", edgecolor="#4c4c4c"
        )
        ax.add_patch(box)
        ax.text(origin_x + width/2, 4.8, title,
                ha="center", fontsize=13, fontweight="bold")

        # 均匀分布 gates
        step = height / (len(gates)+1)
        for idx, (gate, color) in enumerate(zip(gates, colors)):
            y = 4.2 - idx*step
            rect = patches.Rectangle(
                (origin_x + 0.5, y-0.2),
                width - 1.0, 0.6,
                linewidth=1.0, facecolor=color, edgecolor=color, alpha=0.25
            )
            ax.add_patch(rect)
            ax.text(origin_x + width/2, y+0.1, gate,
                    ha="center", fontsize=10, fontweight="medium")

        # 输入/输出文字 & 箭头，留足空隙
        ax.text(origin_x - 0.7, 3.3, "$x_t$", ha="right", fontsize=11)
        ax.annotate("", xy=(origin_x, 3.3), xytext=(origin_x-0.6, 3.3),
                    arrowprops=dict(arrowstyle="->", linewidth=1.3))

        ax.text(origin_x - 0.7, 2.4, "$h_{t-1}$", ha="right", fontsize=11)
        ax.annotate("", xy=(origin_x, 2.4), xytext=(origin_x-0.6, 2.4),
                    arrowprops=dict(arrowstyle="->", linewidth=1.3))

        ax.text(origin_x + width + 0.7, 3.3, "$h_t$", ha="left", fontsize=11)
        ax.annotate("", xy=(origin_x+width+0.6, 3.3), xytext=(origin_x+width, 3.3),
                    arrowprops=dict(arrowstyle="->", linewidth=1.3))

        # LSTM 专属 cell state
        if show_c:
            ax.annotate("$c_t$", xy=(origin_x+width/2, 1.2),
                        ha="center", fontsize=11, color="#555555")
            ax.annotate("", xy=(origin_x+width/2, 1.2), xytext=(origin_x+width/2, 4.5),
                        arrowprops=dict(arrowstyle="->", linestyle="--", color="#555555"))

    # Gate 配置
    lstm_gates = ["Forget gate", "Input gate", "Candidate cell", "Output gate"]
    lstm_colors = ["#5DADE2", "#F5B041", "#58D68D", "#EC7063"]
    gru_gates = ["Update gate", "Reset gate", "Candidate state"]
    gru_colors = ["#5DADE2", "#F5B041", "#58D68D"]

    draw_cell(0.8, "LSTM cell", lstm_gates, lstm_colors, show_c=True)
    draw_cell(7.0, "GRU cell", gru_gates, gru_colors)

    ax.text(6.0, 5.3, "Gate comparison", ha="center", fontsize=15, fontweight="bold")
    ax.set_xlim(-2, 13)
    ax.set_ylim(0.5, 6)

    fig.savefig(out_dir / "lstm_gru_gate_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rnn_applications(out_dir: Path) -> None:
    """Overview of RNN applications"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    sections = [
        ("Time-series Forecasting", ["Sensors", "Encoder RNN", "Decoder", "Forecast"]),
        ("Language Modeling", ["Embedding", "Stacked LSTM", "Softmax", "Generated text"]),
        ("Speech Modeling", ["Acoustic features", "Bi-GRU", "CTC / Attn", "Transcript / Audio"]),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, ((title, stages), color) in enumerate(zip(sections, colors)):
        x0 = idx * 3.8
        box = patches.FancyBboxPatch(
            (x0, 0.6), 3.2, 3.5,
            boxstyle="round,pad=0.3", linewidth=1.6,
            edgecolor=color, facecolor=color, alpha=0.15
        )
        ax.add_patch(box)
        ax.text(x0 + 1.6, 3.9, title, ha="center", fontsize=12, fontweight="bold", color="#333333")

        for stage_idx, stage in enumerate(stages):
            ax.text(x0 + 1.6, 3.2 - stage_idx * 0.7,
                    stage, ha="center", fontsize=10, color="#333333")

    ax.text(5.8, 4.6, "RNN application landscape", ha="center", fontsize=15, fontweight="bold")
    ax.set_xlim(-1, 13)
    ax.set_ylim(0, 5.2)

    fig.savefig(out_dir / "rnn_applications_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = figures_dir()
    plot_rnn_unrolled(out_dir)
    plot_lstm_gru_comparison(out_dir)
    plot_rnn_applications(out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
