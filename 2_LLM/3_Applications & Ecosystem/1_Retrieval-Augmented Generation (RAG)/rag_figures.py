import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle


def draw_rag_pipeline(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")
    stages = [
        (0.08, "#1f77b4", "#e3f2fd", "User Query\n(Natural Language)"),
        (0.3, "#2ca02c", "#e8f8f5", "Embedding\nEncoder"),
        (0.52, "#ff7f0e", "#fef5e6", "Vector Search\nTop-k Retrieval"),
        (0.74, "#9467bd", "#f5eafc", "Context Fusion\n(Rewrite/Rank)"),
        (0.92, "#d62728", "#fdecea", "LLM Generation\nGrounded Answer"),
    ]

    for x, edge, face, label in stages:
        box = FancyBboxPatch(
            (x - 0.12, 0.32),
            0.24,
            0.34,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, 0.49, label, ha="center", va="center", fontsize=11, color="#2c3e50")

    for idx in range(len(stages) - 1):
        start = stages[idx][0] + 0.12
        end = stages[idx + 1][0] - 0.12
        arrow = FancyArrowPatch(
            (start, 0.49),
            (end, 0.49),
            arrowstyle=ArrowStyle("Simple", head_length=12, head_width=10),
            linewidth=1.2,
            color="#2c3e50",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.82,
        "Retrieval-Augmented Generation Pipeline",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.16,
        "Queries are embedded, matched via vector search, fused with evidence, and fed into the LLM for grounded responses.",
        ha="center",
        fontsize=10,
        color="#34495e",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_chunking_embedding(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axis("off")

    doc_box = FancyBboxPatch(
        (0.05, 0.25),
        0.26,
        0.5,
        boxstyle="round,pad=0.04",
        linewidth=1.4,
        edgecolor="#17becf",
        facecolor="#e8f8fc",
    )
    ax.add_patch(doc_box)
    ax.text(0.18, 0.67, "Source Document\n(HTML/PDF/SQL)", ha="center", va="center", fontsize=11, color="#2c3e50")
    ax.text(0.18, 0.42, "Chunker\nSliding Window / Semantic Split", ha="center", va="center", fontsize=10, color="#2c3e50")

    chunk_positions = [0.36, 0.48, 0.6]
    for idx, y in enumerate(chunk_positions):
        chunk_box = FancyBboxPatch(
            (0.32, y - 0.07),
            0.18,
            0.14,
            boxstyle="round,pad=0.03",
            linewidth=1.3,
            edgecolor="#ffbb78",
            facecolor="#fff2e1",
        )
        ax.add_patch(chunk_box)
        ax.text(0.41, y, f"Chunk {idx+1}", ha="center", va="center", fontsize=10, color="#2c3e50")

        arrow = FancyArrowPatch(
            (0.29, 0.45),
            (0.32, y),
            arrowstyle=ArrowStyle("Simple", head_length=10, head_width=8),
            linewidth=1.0,
            color="#34495e",
            connectionstyle=f"arc3,rad={(idx - 1) * 0.2}",
        )
        ax.add_patch(arrow)

    for idx, y in enumerate(chunk_positions):
        embed_box = FancyBboxPatch(
            (0.55, y - 0.07),
            0.18,
            0.14,
            boxstyle="round,pad=0.03",
            linewidth=1.3,
            edgecolor="#98df8a",
            facecolor="#edf7ec",
        )
        ax.add_patch(embed_box)
        ax.text(0.64, y, f"Embedding {idx+1}", ha="center", va="center", fontsize=10, color="#2c3e50")

        arrow = FancyArrowPatch(
            (0.5, y),
            (0.55, y),
            arrowstyle=ArrowStyle("Simple", head_length=10, head_width=8),
            linewidth=1.0,
            color="#34495e",
        )
        ax.add_patch(arrow)

    store_box = FancyBboxPatch(
        (0.78, 0.25),
        0.18,
        0.5,
        boxstyle="round,pad=0.04",
        linewidth=1.4,
        edgecolor="#c5b0d5",
        facecolor="#f6f0fa",
    )
    ax.add_patch(store_box)
    ax.text(0.87, 0.63, "Vector Store\n(FAISS/Milvus/\nChroma)", ha="center", va="center", fontsize=11, color="#2c3e50")
    ax.text(0.87, 0.39, "Metadata Index\n+ Filters", ha="center", va="center", fontsize=10, color="#2c3e50")

    ax.text(
        0.5,
        0.88,
        "Chunking and Embedding Workflow",
        ha="center",
        fontsize=14,
        color="#2c3e50",
        fontweight="bold",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_rag_pipeline("figures/rag_pipeline.png")
    draw_chunking_embedding("figures/chunk_embedding.png")


if __name__ == "__main__":
    main()
