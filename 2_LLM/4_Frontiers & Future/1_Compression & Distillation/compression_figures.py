import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_compression_landscape(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")

    blocks = [
        (0.1, "#1f77b4", "#e3f2fd", "Baseline LLM\n(full precision)"),
        (0.37, "#ff7f0e", "#fef5e6", "Pruning\n(structured / unstructured)"),
        (0.64, "#2ca02c", "#e8f8f5", "Quantization\n(INT8, INT4, FP8)"),
        (0.9, "#9467bd", "#f5eafc", "Distilled Model\n(student)"),
    ]

    for x, edge, face, label in blocks:
        box = FancyBboxPatch(
            (x - 0.14, 0.33),
            0.28,
            0.34,
            boxstyle="round,pad=0.03",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    arrow_pairs = [
        ((blocks[i][0] + 0.14, 0.5), (blocks[i + 1][0] - 0.14, 0.5))
        for i in range(len(blocks) - 1)
    ]

    for (sx, sy), (tx, ty) in arrow_pairs:
        arrow = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            arrowstyle=ArrowStyle("Simple", head_length=12, head_width=10),
            linewidth=1.1,
            color="#34495e",
        )
        ax.add_patch(arrow)

    annotations = [
        (0.24, 0.65, "Sparsity masks,\nlottery ticket"),
        (0.51, 0.65, "Weight-only,\nactivation-aware"),
        (0.78, 0.65, "Teacher guidance,\nKD loss"),
    ]

    for x, y, text in annotations:
        ax.text(x, y, text, ha="center", va="center", fontsize=10, color="#34495e")

    ax.text(
        0.5,
        0.82,
        "Compression roadmap from dense LLM to compact deployment",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.16,
        "Pruning removes redundancy, quantization reduces precision, distillation transfers knowledge to smaller students.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_edge_deployment(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")

    layers = [
        (0.15, 0.7, 0.28, 0.18, "#1f77b4", "#e3f2fd", "Model Zoo &\nCompression Pipeline"),
        (0.15, 0.42, 0.28, 0.18, "#ff7f0e", "#fef5e6", "TinyLLM Runtime\n(Quantized kernels)"),
        (0.15, 0.14, 0.28, 0.18, "#2ca02c", "#e8f8f5", "Monitoring &\nA/B Feedback"),
        (0.52, 0.7, 0.28, 0.18, "#9467bd", "#f5eafc", "Edge Hardware\n(ARM, GPU, NPU)"),
        (0.52, 0.42, 0.28, 0.18, "#d62728", "#fdecea", "On-device Services\n(NLP, vision, speech)"),
        (0.52, 0.14, 0.28, 0.18, "#17becf", "#e8f8fc", "Telemetry\n(latency, energy)"),
        (0.82, 0.56, 0.26, 0.32, "#8c564b", "#f7ebe1", "Orchestrator\n(model updates, rollout)"),
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

    arrows = [
        ((0.29, 0.7), (0.29, 0.6)),
        ((0.29, 0.42), (0.29, 0.32)),
        ((0.66, 0.7), (0.66, 0.6)),
        ((0.66, 0.42), (0.66, 0.32)),
        ((0.66, 0.32), (0.66, 0.22)),
        ((0.4, 0.51), (0.52, 0.51)),
        ((0.8, 0.56), (0.8, 0.4)),
        ((0.8, 0.4), (0.66, 0.4)),
        ((0.8, 0.67), (0.4, 0.74)),
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
        0.93,
        "TinyLLM edge deployment stack",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.05,
        "Compressed models flow through quantized runtimes onto heterogeneous edge hardware; orchestration closes the loop with monitoring.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_compression_landscape("figures/compression_landscape.png")
    draw_edge_deployment("figures/tinyllm_edge_stack.png")


if __name__ == "__main__":
    main()

