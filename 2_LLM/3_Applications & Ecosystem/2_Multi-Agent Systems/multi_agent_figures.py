import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_agent_architecture(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axis("off")

    layers = [
        (0.2, 0.7, 0.6, 0.4, "#1f77b4", "#e3f2fd", "Planner\n(Task decomposition, policy selection)"),
        (0.2, 0.45, 0.28, 0.18, "#ff7f0e", "#fef5e6", "Executor\n(Action calls)"),
        (0.52, 0.45, 0.28, 0.18, "#2ca02c", "#e8f8f5", "Memory\n(Working + long-term)"),
        (0.2, 0.2, 0.28, 0.18, "#9467bd", "#f5eafc", "Tooling Layer\n(API, simulation, code)"),
        (0.52, 0.2, 0.28, 0.18, "#d62728", "#fdecea", "Environment\n(Data, humans, services)"),
    ]

    for x, y, w, h, edge, face, label in layers:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    arrows = [
        ((0.5, 0.7), (0.34, 0.63)),
        ((0.5, 0.7), (0.66, 0.63)),
        ((0.34, 0.45), (0.34, 0.38)),
        ((0.66, 0.45), (0.66, 0.38)),
        ((0.34, 0.2), (0.66, 0.2)),
        ((0.66, 0.38), (0.66, 0.31)),
    ]

    for (sx, sy), (tx, ty) in arrows:
        arrow = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            arrowstyle=ArrowStyle("Simple", head_length=10, head_width=8),
            linewidth=1.1,
            color="#34495e",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.92,
        "Multi-agent architecture: planner, executor, memory, tools, environment",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )

    ax.text(
        0.5,
        0.06,
        "Signals flow from planner to execution, while memory and tooling provide feedback loops to the environment.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_collaboration_flow(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axis("off")

    agents = [
        (0.15, 0.65, "#1f77b4", "#e3f2fd", "Research Agent\n(dataset scouting)"),
        (0.5, 0.65, "#ff7f0e", "#fef5e6", "Code Agent\n(pipeline build)"),
        (0.85, 0.65, "#2ca02c", "#e8f8f5", "Analysis Agent\n(model eval)"),
        (0.32, 0.3, "#9467bd", "#f5eafc", "Memory Hub\n(vector DB, logs)"),
        (0.68, 0.3, "#d62728", "#fdecea", "Orchestrator\n(planner, critic)"),
    ]

    for x, y, edge, face, label in agents:
        box = FancyBboxPatch(
            (x - 0.12, y - 0.1),
            0.24,
            0.2,
            boxstyle="round,pad=0.04",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    arrows = [
        ((0.27, 0.55), (0.38, 0.38)),
        ((0.5, 0.55), (0.38, 0.38)),
        ((0.73, 0.55), (0.38, 0.38)),
        ((0.38, 0.3), (0.62, 0.3)),
        ((0.62, 0.42), (0.27, 0.55)),
        ((0.62, 0.42), (0.5, 0.55)),
        ((0.62, 0.42), (0.73, 0.55)),
        ((0.62, 0.18), (0.5, 0.18)),
    ]

    for (sx, sy), (tx, ty) in arrows:
        arrow = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            arrowstyle=ArrowStyle("Simple", head_length=10, head_width=8),
            linewidth=1.0,
            color="#34495e",
            connectionstyle="arc3,rad=0.05",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.92,
        "Collaborative workflow across specialized agents",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )

    ax.text(
        0.5,
        0.08,
        "Planner orchestrates research, coding, and analysis agents; shared memory keeps artifacts synchronized.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_agent_architecture("figures/multi_agent_architecture.png")
    draw_collaboration_flow("figures/agent_collaboration.png")


if __name__ == "__main__":
    main()

