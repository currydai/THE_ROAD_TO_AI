import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_multimodal_pipeline(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.axis("off")

    modules = [
        (0.08, "#1f77b4", "#e3f2fd", "Text Encoder\n(Transformer)"),
        (0.3, "#ff7f0e", "#fef5e6", "Vision Encoder\n(CLIP/ViT)"),
        (0.52, "#2ca02c", "#e8f8f5", "Audio/Video Encoder\n(Whisper, VideoMAE)"),
        (0.74, "#9467bd", "#f5eafc", "Fusion Layer\n(Co-attention, Q-Former)"),
        (0.92, "#d62728", "#fdecea", "LLM Decoder\n(GPT-4V, LLaVA)"),
    ]

    for x, edge, face, label in modules:
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

    for idx in range(len(modules) - 1):
        start = modules[idx][0] + 0.12
        end = modules[idx + 1][0] - 0.12
        arrow = FancyArrowPatch(
            (start, 0.5),
            (end, 0.5),
            arrowstyle=ArrowStyle("Simple", head_length=12, head_width=10),
            linewidth=1.1,
            color="#34495e",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.82,
        "Multimodal perception and fusion pipeline",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.16,
        "Separate encoders extract modality features; fusion and adapters align representations before LLM decoding.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_alignment_adapters(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axis("off")

    blocks = [
        (0.15, 0.65, "#1f77b4", "#e3f2fd", "Modality Encoders\n(Text, Image, Audio)"),
        (0.5, 0.65, "#ff7f0e", "#fef5e6", "Projection Heads\n(linear, MLP, LoRA)"),
        (0.85, 0.65, "#2ca02c", "#e8f8f5", "Shared Latent Space\n(token-aligned)"),
        (0.15, 0.3, "#9467bd", "#f5eafc", "Temporal Aligners\n(CTC, attention)"),
        (0.5, 0.3, "#d62728", "#fdecea", "Cross-modal Adapters\n(Q-Former, Perceiver)"),
        (0.85, 0.3, "#8c564b", "#f7ebe1", "Task Heads\n(VQA, captioning)"),
    ]

    for x, y, edge, face, label in blocks:
        box = FancyBboxPatch(
            (x - 0.12, y - 0.1),
            0.24,
            0.2,
            boxstyle="round,pad=0.035",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    arrows = [
        ((0.27, 0.65), (0.38, 0.65)),
        ((0.62, 0.65), (0.73, 0.65)),
        ((0.27, 0.3), (0.38, 0.3)),
        ((0.62, 0.3), (0.73, 0.3)),
        ((0.5, 0.55), (0.5, 0.4)),
        ((0.73, 0.65), (0.73, 0.4)),
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
        "Alignment adapters for multimodal fusion",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "Projection heads and cross-modal adapters map heterogeneous signals into a shared latent space before task-specific decoding.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_multimodal_pipeline("figures/multimodal_pipeline.png")
    draw_alignment_adapters("figures/alignment_adapters.png")


if __name__ == "__main__":
    main()

