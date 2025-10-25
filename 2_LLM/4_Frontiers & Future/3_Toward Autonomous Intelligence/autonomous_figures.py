import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_world_model_pipeline(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.axis("off")

    stages = [
        (0.1, "#1f77b4", "#e3f2fd", "Sensory Streams\n(text, vision, telemetry)"),
        (0.34, "#ff7f0e", "#fef5e6", "World Model\n(latent dynamics)"),
        (0.58, "#2ca02c", "#e8f8f5", "Memory Bank\n(Episodic + Semantic)"),
        (0.82, "#9467bd", "#f5eafc", "Policy / LM\n(planning, dialogue)"),
    ]

    for x, edge, face, label in stages:
        box = FancyBboxPatch(
            (x - 0.14, 0.35),
            0.28,
            0.32,
            boxstyle="round,pad=0.03",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    arrows = [
        ((0.24, 0.5), (0.2, 0.64)),
        ((0.24, 0.5), (0.28, 0.36)),
        ((0.48, 0.5), (0.52, 0.5)),
        ((0.72, 0.5), (0.76, 0.5)),
        ((0.58, 0.67), (0.58, 0.8)),
        ((0.82, 0.67), (0.82, 0.8)),
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

    ax.text(0.58, 0.86, "Prediction & imagination rollouts", ha="center", fontsize=10, color="#34495e")
    ax.text(0.82, 0.86, "Action selection & tool use", ha="center", fontsize=10, color="#34495e")

    ax.text(
        0.5,
        0.82,
        "World-model and memory-augmented language model pipeline",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.2,
        "Raw sensory data feeds latent world models; memories consolidate episodes; policies leverage both for planning and dialogue.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_agentic_ecosystem(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")

    nodes = [
        (0.18, 0.72, "#1f77b4", "#e3f2fd", "MetaGPT\n(Role planners)"),
        (0.5, 0.72, "#ff7f0e", "#fef5e6", "Voyager\n(Open-world agent)"),
        (0.82, 0.72, "#2ca02c", "#e8f8f5", "Self-Improving Loop\n(Eval, retrain)"),
        (0.18, 0.32, "#9467bd", "#f5eafc", "Memory Infra\n(Vector DB, logs)"),
        (0.5, 0.32, "#d62728", "#fdecea", "Continual Learning\n(Replay, adapters)"),
        (0.82, 0.32, "#17becf", "#e8f8fc", "Science/Materials/\nGeophysics Apps"),
    ]

    for x, y, edge, face, label in nodes:
        box = FancyBboxPatch(
            (x - 0.14, y - 0.1),
            0.28,
            0.2,
            boxstyle="round,pad=0.035",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    links = [
        ((0.32, 0.72), (0.36, 0.72)),
        ((0.64, 0.72), (0.68, 0.72)),
        ((0.18, 0.62), (0.18, 0.42)),
        ((0.5, 0.62), (0.5, 0.42)),
        ((0.82, 0.62), (0.82, 0.42)),
        ((0.32, 0.32), (0.36, 0.32)),
        ((0.64, 0.32), (0.68, 0.32)),
        ((0.36, 0.32), (0.5, 0.5)),
        ((0.64, 0.32), (0.5, 0.5)),
        ((0.5, 0.32), (0.5, 0.18)),
    ]

    for (sx, sy), (tx, ty) in links:
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
        "Agentic ecosystem: planners, explorers, continual learning, and scientific deployment",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "MetaGPT orchestrates teams; Voyager gathers experiences; continual learning and memory feed back into scientific applications.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_world_model_pipeline("figures/world_model_pipeline.png")
    draw_agentic_ecosystem("figures/agentic_ecosystem.png")


if __name__ == "__main__":
    main()

