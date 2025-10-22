"""
Generate figures for the deep reinforcement learning tutorials.

Outputs saved under figures/:
  - rl_overview.png
  - alphago_pipeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.grid": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def plot_rl_overview() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")

    boxes = [
        (0.3, 2.4, "Value-Based\n(DQN)", "#1f77b4"),
        (4.1, 2.4, "Policy Gradient\n(REINFORCE / PPO)", "#ff7f0e"),
        (7.9, 2.4, "Actor-Critic\n(A2C / SAC)", "#2ca02c"),
    ]

    descriptions = [
        ["Replay buffer", "Target network", "Double/Dueling variants", "Discrete actions"],
        ["On-policy updates", "Stochastic policies", "GAE, clipping", "Continuous actions"],
        ["Actor + critic networks", "Entropy regularization", "Off-policy (SAC)", "Stability via bootstrapping"],
    ]

    for (x, y, title, color), desc in zip(boxes, descriptions):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                2.2,
                1.5,
                boxstyle="round,pad=0.3",
                linewidth=1.5,
                edgecolor=color,
                facecolor=(0.94, 0.97, 1.0),
            )
        )
        ax.text(x + 1.1, y + 1.2, title, ha="center", va="center", fontsize=12, color=color, fontweight="bold")
        for idx, line in enumerate(desc):
            ax.text(x + 0.2, y + 0.9 - idx * 0.3, f"• {line}", ha="left", va="center", fontsize=10)

    arrows = [
        ((2.5, 2.7), (3.9, 2.7), "Policy directly optimized"),
        ((6.3, 2.7), (7.7, 2.7), "Critic reduces variance"),
    ]
    for (x0, y0), (x1, y1), text in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#555555"))
        ax.text((x0 + x1) / 2, y0 + 0.4, text, ha="center", fontsize=10)

    ax.text(
        5.5,
        0.8,
        "Key considerations:\n• Sample efficiency\n• Stability (variance vs bias)\n• Action space (discrete vs continuous)\n• On-policy vs off-policy trade-offs",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8", edgecolor="#9467bd"),
    )

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.set_title("Comparison of DRL Algorithm Families")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rl_overview.png", dpi=300)
    plt.close(fig)


def plot_alphago_pipeline() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")

    stages = [
        (0.3, 2.5, "Expert Games", "#1f77b4"),
        (2.1, 2.5, "Supervised Policy\n$p_{\theta}$", "#1f77b4"),
        (3.9, 2.5, "Self-Play RL\n$p_{\theta}'$", "#ff7f0e"),
        (5.7, 2.5, "Value Network\n$v_{\phi}$", "#2ca02c"),
        (7.5, 2.5, "Monte Carlo Tree Search", "#9467bd"),
        (9.3, 2.5, "Move Selection", "#d62728"),
    ]

    for x, y, text, color in stages:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                1.6,
                1.4,
                boxstyle="round,pad=0.3",
                linewidth=1.5,
                edgecolor=color,
                facecolor=(0.94, 0.97, 1.0),
            )
        )
        ax.text(x + 0.8, y + 0.7, text, ha="center", va="center", fontsize=11, color=color)

    def arrow(x0: float, y0: float, x1: float, y1: float, text: str | None = None) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#555555"))
        if text:
            ax.text((x0 + x1) / 2, y0 + 0.45, text, ha="center", fontsize=10)

    arrow(1.9, 3.2, 2.1, 3.2, "SL training")
    arrow(3.7, 3.2, 3.9, 3.2, "Policy gradient")
    arrow(5.5, 3.2, 5.7, 3.2, "Self-play outcomes")
    arrow(7.3, 3.2, 7.5, 3.2, "Policy/value priors")
    arrow(9.1, 3.2, 9.3, 3.2, "Best move")

    ax.text(
        3.0,
        1.1,
        "Replay buffer of self-play games\nValue targets from Monte Carlo outcomes\nParallel search workers",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ff7f0e"),
        fontsize=10,
    )

    ax.text(
        8.2,
        1.0,
        "PUCT selection:\n$Q(s,a) + c_{\\mathrm{puct}} P(s,a) \\frac{\\sqrt{\\sum_b N(s,b)}}{1 + N(s,a)}$\nMix of policy priors and visit counts",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#9467bd"),
        fontsize=10,
    )

    ax.set_xlim(0, 11)
    ax.set_ylim(0.5, 4.2)
    ax.set_title("AlphaGo Training and Decision Pipeline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alphago_pipeline.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_rl_overview()
    plot_alphago_pipeline()


if __name__ == "__main__":
    main()
