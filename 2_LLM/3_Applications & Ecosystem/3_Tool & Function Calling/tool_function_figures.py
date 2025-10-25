import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_function_calling(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")

    stages = [
        (0.08, "#1f77b4", "#e3f2fd", "User\nRequest"),
        (0.28, "#2ca02c", "#e8f8f5", "LLM\nParser"),
        (0.48, "#ff7f0e", "#fef5e6", "Function\nRouter"),
        (0.68, "#9467bd", "#f5eafc", "Tool Runtime\n(API / DB / Compute)"),
        (0.88, "#d62728", "#fdecea", "Structured\nResponse"),
    ]

    for x, edge, face, label in stages:
        box = FancyBboxPatch(
            (x - 0.12, 0.33),
            0.24,
            0.34,
            boxstyle="round,pad=0.03",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    arrows = []
    for idx in range(len(stages) - 1):
        start = stages[idx][0] + 0.12
        end = stages[idx + 1][0] - 0.12
        arrows.append(((start, 0.5), (end, 0.5)))

    for (sx, sy), (tx, ty) in arrows:
        arrow = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            arrowstyle=ArrowStyle("Simple", head_length=12, head_width=10),
            linewidth=1.1,
            color="#34495e",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.82,
        "Function calling pipeline with routing and tool execution",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.16,
        "LLM selects JSON arguments, router validates schema, tool runtime executes and streams results.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_agent_stack(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")

    layers = [
        (0.08, 0.7, 0.28, 0.18, "#1f77b4", "#e3f2fd", "WebAgent\nController"),
        (0.08, 0.42, 0.28, 0.18, "#ff7f0e", "#fef5e6", "Browser Tooling\n(Playwright, DOM API)"),
        (0.08, 0.14, 0.28, 0.18, "#2ca02c", "#e8f8f5", "State Cache\n(Session, Cookies)"),
        (0.38, 0.7, 0.28, 0.18, "#9467bd", "#f5eafc", "OS-Agent\nTask Manager"),
        (0.38, 0.42, 0.28, 0.18, "#d62728", "#fdecea", "System Tools\n(Shell, File I/O, Apps)"),
        (0.38, 0.14, 0.28, 0.18, "#17becf", "#e8f8fc", "Device Context\n(Processes, Logs)"),
        (0.68, 0.57, 0.26, 0.32, "#8c564b", "#f7ebe1", "Shared Memory Hub\n(Vector DB, Timeline)"),
        (0.68, 0.2, 0.26, 0.18, "#7f7f7f", "#f0f0f0", "Governance Layer\n(Auth, Policy, Audit)"),
    ]

    for x, y, w, h, edge, face, label in layers:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    connections = [
        ((0.22, 0.7), (0.22, 0.6)),
        ((0.22, 0.42), (0.22, 0.32)),
        ((0.52, 0.7), (0.52, 0.6)),
        ((0.52, 0.42), (0.52, 0.32)),
        ((0.34, 0.62), (0.68, 0.62)),
        ((0.34, 0.22), (0.68, 0.22)),
        ((0.81, 0.57), (0.81, 0.38)),
    ]

    for (sx, sy), (tx, ty) in connections:
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
        0.93,
        "Hybrid WebAgent / OS-Agent stack with shared memory and governance",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )

    ax.text(
        0.5,
        0.05,
        "Controllers orchestrate browser and OS tooling; shared memory synchronizes artifacts while governance enforces policies.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_function_calling("figures/function_calling_pipeline.png")
    draw_agent_stack("figures/hybrid_agent_stack.png")


if __name__ == "__main__":
    main()

