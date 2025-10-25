import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch


def draw_hf_pipeline(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    stages = [
        (0.05, "Data\nCuration\n(Datasets)"),
        (0.27, "Tokenizer\n& Feature\nEngineering"),
        (0.49, "Trainer\nConfiguration\n(TrainingArguments)"),
        (0.72, "Accelerators\n(Accelerate/\nDeepSpeed)"),
        (0.9, "Evaluation\n& Push to\nHub/Registry"),
    ]

    for x, label in stages:
        box = FancyBboxPatch(
            (x - 0.08, 0.35),
            0.16,
            0.3,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#1f77b4",
            facecolor="#d6eaf8",
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=11)

    for idx in range(len(stages) - 1):
        start = stages[idx][0] + 0.08
        end = stages[idx + 1][0] - 0.08
        arrow = FancyArrowPatch(
            (start, 0.5),
            (end, 0.5),
            arrowstyle=ArrowStyle("Simple", head_length=12, head_width=12),
            linewidth=1.2,
            color="#2c3e50",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.82,
        "Hugging Face Transformers Training Flow",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.18,
        "Iterative experimentation loop with checkpoints, metrics, and model registry integration.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_distributed_landscape(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    columns = [
        (0.18, "#27ae60", "#eafaf1", "DeepSpeed", ["ZeRO-1/2/3", "Pipeline/TP", "CPU Offload", "Inference Kernels"]),
        (0.5, "#8e44ad", "#f5eafc", "Megatron-LM", ["Tensor Parallel", "Sequence Parallel", "MoE Support", "FP8 Kernels"]),
        (0.82, "#c0392b", "#fdecea", "Colossal-AI", ["Hybrid Parallel", "Booster API", "Galvatron AutoTP", "Gemini Memory Mgmt"]),
    ]

    for x, edge_color, face_color, title, bullets in columns:
        box = FancyBboxPatch(
            (x - 0.13, 0.3),
            0.26,
            0.45,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor=edge_color,
            facecolor=face_color,
        )
        ax.add_patch(box)
        ax.text(x, 0.68, title, ha="center", va="center", fontsize=13, color=edge_color, fontweight="bold")
        for idx, bullet in enumerate(bullets):
            ax.text(
                x,
                0.55 - idx * 0.1,
                f"• {bullet}",
                ha="center",
                va="center",
                fontsize=10,
                color="#2c3e50",
            )

    ax.text(
        0.5,
        0.88,
        "Distributed Training Framework Capabilities",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_hf_pipeline("figures/hf_training_pipeline.png")
    draw_distributed_landscape("figures/distributed_frameworks.png")


if __name__ == "__main__":
    main()

