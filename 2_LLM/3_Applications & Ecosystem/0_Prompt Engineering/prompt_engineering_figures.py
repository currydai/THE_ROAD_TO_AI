import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_prompt_modes(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    stages = [
        (0.1, "#1f77b4", "#e3f2fd", "Zero-shot\nPattern Recall"),
        (0.38, "#2ca02c", "#e8f8f5", "Few-shot\nDemonstrations"),
        (0.66, "#ff7f0e", "#fef5e6", "Chain-of-Thought\nDecomposition"),
        (0.9, "#9467bd", "#f5eafc", "Verification &\nRefinement"),
    ]

    for x, edge, face, label in stages:
        box = FancyBboxPatch(
            (x - 0.13, 0.3),
            0.26,
            0.32,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, 0.46, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    for idx in range(len(stages) - 1):
        start = stages[idx][0] + 0.13
        end = stages[idx + 1][0] - 0.13
        arrow = FancyArrowPatch(
            (start, 0.46),
            (end, 0.46),
            arrowstyle=ArrowStyle("Simple", head_length=12, head_width=10),
            linewidth=1.2,
            color="#2c3e50",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.8,
        "Prompting Modes for Reasoning Tasks",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.15,
        "Progressively enrich prompts from zero knowledge to structured reasoning and self-checking.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_reasoning_stacks(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    nodes = [
        (0.2, 0.75, "#17becf", "#e8f8fc", "System Prompt\nContext"),
        (0.2, 0.45, "#ff9896", "#fdecea", "State\nMemory"),
        (0.5, 0.6, "#98df8a", "#edf7ec", "ReAct Loop"),
        (0.8, 0.6, "#c5b0d5", "#f6f0fa", "Tree-of-Thoughts\nPlanner"),
        (0.5, 0.25, "#aec7e8", "#eef3fc", "External\nTools / APIs"),
        (0.8, 0.25, "#ffbb78", "#fff2e1", "Evaluation &\nScoring"),
    ]

    for x, y, edge, face, label in nodes:
        box = FancyBboxPatch(
            (x - 0.12, y - 0.1),
            0.24,
            0.2,
            boxstyle="round,pad=0.03",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    edges = [
        (0.2, 0.65, 0.5, 0.68),
        (0.2, 0.35, 0.5, 0.52),
        (0.5, 0.5, 0.8, 0.5),
        (0.5, 0.5, 0.5, 0.35),
        (0.8, 0.5, 0.8, 0.35),
    ]

    for sx, sy, tx, ty in edges:
        arrow = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            arrowstyle=ArrowStyle("Simple", head_length=10, head_width=8),
            linewidth=1.1,
            color="#34495e",
            connectionstyle="arc3,rad=0.05",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.92,
        "Reasoning-Augmented Prompting Stack",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "System instructions ground the agent, ReAct handles iterative reasoning, Tree-of-Thoughts explores branches,\n"
        "while tool usage and evaluators refine outputs under token constraints.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_prompt_modes("figures/prompt_modes.png")
    draw_reasoning_stacks("figures/reasoning_stacks.png")


if __name__ == "__main__":
    main()

