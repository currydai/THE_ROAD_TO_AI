import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_benchmark_landscape(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.axis("off")

    benchmarks = [
        (0.18, "#1f77b4", "#e3f2fd", "MMLU\n57 subjects"),
        (0.5, "#ff7f0e", "#fef5e6", "GSM8K\nGrade-school math"),
        (0.82, "#2ca02c", "#e8f8f5", "BIG-Bench\n204 tasks"),
    ]

    for x, edge, face, label in benchmarks:
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
        ((0.32, 0.5), (0.36, 0.5)),
        ((0.64, 0.5), (0.68, 0.5)),
    ]

    for (sx, sy), (tx, ty) in arrows:
        arrow = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            arrowstyle=ArrowStyle("Simple", head_length=12, head_width=10),
            linewidth=1.1,
            color="#34495e",
        )
        ax.add_patch(arrow)

    ax.text(0.18, 0.68, "Knowledge\n+ reasoning", ha="center", va="center", fontsize=10, color="#34495e")
    ax.text(0.5, 0.68, "Multi-step\nreasoning", ha="center", va="center", fontsize=10, color="#34495e")
    ax.text(0.82, 0.68, "Broad capabilities\n+ safety probes", ha="center", va="center", fontsize=10, color="#34495e")

    ax.text(
        0.5,
        0.85,
        "Benchmark spectrum for LLM evaluation",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.2,
        "MMLU measures academic knowledge, GSM8K targets mathematical reasoning, BIG-Bench samples long-tail abilities.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_interpretability_pipeline(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")

    blocks = [
        (0.1, 0.7, 0.26, 0.18, "#1f77b4", "#e3f2fd", "Inputs & Prompt\n(text, multimodal)"),
        (0.1, 0.4, 0.26, 0.18, "#ff7f0e", "#fef5e6", "Attention Probes\n(heads, layers)"),
        (0.1, 0.1, 0.26, 0.18, "#2ca02c", "#e8f8f5", "Attribution Scores\n(SHAP, IG)"),
        (0.42, 0.55, 0.26, 0.18, "#9467bd", "#f5eafc", "Visualization\n(heatmaps, graphs)"),
        (0.74, 0.55, 0.26, 0.18, "#d62728", "#fdecea", "Insights\n(bias, safety, control)"),
    ]

    for x, y, w, h, edge, face, label in blocks:
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

    arrows = [
        ((0.23, 0.7), (0.23, 0.58)),
        ((0.23, 0.4), (0.23, 0.28)),
        ((0.36, 0.58), (0.42, 0.59)),
        ((0.36, 0.28), (0.42, 0.51)),
        ((0.68, 0.64), (0.74, 0.64)),
        ((0.68, 0.46), (0.74, 0.46)),
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
        0.9,
        "Interpretability pipeline: attention and attribution analysis",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.05,
        "Attention heads reveal token focus; attribution scores quantify feature influence; visualization turns findings into actionable insights.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_benchmark_landscape("figures/benchmark_landscape.png")
    draw_interpretability_pipeline("figures/interpretability_pipeline.png")


if __name__ == "__main__":
    main()

