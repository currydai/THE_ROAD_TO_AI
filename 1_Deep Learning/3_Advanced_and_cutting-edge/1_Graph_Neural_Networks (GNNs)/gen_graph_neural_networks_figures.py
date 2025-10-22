"""
Generate figures for the graph neural networks tutorials.

Outputs stored in figures/:
  - gcn_graph_structure.png
  - gcn_layer_flow.png
  - gnn_application_landscape.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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


def plot_gcn_graph_structure() -> None:
    _configure_matplotlib()
    G = nx.Graph()
    edges = [
        ("v0", "v1"),
        ("v0", "v2"),
        ("v1", "v2"),
        ("v1", "v3"),
        ("v2", "v4"),
        ("v3", "v4"),
        ("v3", "v5"),
        ("v4", "v5"),
    ]
    G.add_edges_from(edges)

    pos = {
        "v0": (-0.8, 0.4),
        "v1": (-0.2, 0.8),
        "v2": (-0.2, 0.0),
        "v3": (0.4, 0.7),
        "v4": (0.4, -0.1),
        "v5": (1.0, 0.3),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.8, edge_color="#bbbbbb")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#1f77b4", node_size=900, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white")

    ax.scatter([-0.2], [0.0], s=1800, facecolors="none", edgecolors="#ff7f0e", linewidths=2.5)
    ax.text(-0.7, -0.55, "Central node $v_2$\nAggregates normalized neighbors", fontsize=10)

    ax.set_title("GCN Neighborhood Aggregation (Self-loops implied)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gcn_graph_structure.png", dpi=300)
    plt.close(fig)


def plot_gcn_layer_flow() -> None:
    _configure_matplotlib()
    layers = ["Input features", "Layer 1", "Layer 2", "Output logits"]
    x_positions = [0.6, 3.0, 5.4, 7.8]
    y_positions = [2.4, 1.2, 2.4, 1.2, 2.4]
    nodes_per_layer = [5, 5, 5, 3]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    for idx, (layer, x, count) in enumerate(zip(layers, x_positions, nodes_per_layer)):
        ys = np.linspace(0.5, 2.9, count)
        color = "#1f77b4" if idx <= 1 else "#2ca02c" if idx == 2 else "#ff7f0e"
        ax.scatter(np.full(count, x), ys, s=500, c=color, edgecolors="white", linewidths=1.5, zorder=3)
        for y, node_idx in zip(ys, range(count)):
            ax.text(x, y, f"h{idx}{node_idx}", ha="center", va="center", fontsize=9, color="white")
        ax.text(x, 3.2, layer, ha="center", va="center", fontsize=11, color=color)

    # Connective edges between layers
    def connect(x0: float, x1: float, y0: Iterable[float], y1: Iterable[float]) -> None:
        for y_start in y0:
            for y_end in y1:
                ax.plot([x0, x1], [y_start, y_end], color="#cccccc", linewidth=0.8, zorder=1)

    y_inputs = np.linspace(0.5, 2.9, nodes_per_layer[0])
    y_layer1 = np.linspace(0.5, 2.9, nodes_per_layer[1])
    y_layer2 = np.linspace(0.5, 2.9, nodes_per_layer[2])
    y_output = np.linspace(0.8, 2.2, nodes_per_layer[3])

    connect(x_positions[0], x_positions[1], y_inputs, y_layer1)
    connect(x_positions[1], x_positions[2], y_layer1, y_layer2)
    connect(x_positions[2], x_positions[3], y_layer2, y_output)

    ax.text(
        4.2,
        -0.2,
        "Layer 1 aggregates 1-hop context; Layer 2 extends to 2-hop neighborhoods.\nNormalization prevents feature scale explosion.",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gcn_layer_flow.png", dpi=300)
    plt.close(fig)


def plot_gnn_application_landscape() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    domains = [
        ("Social Networks", ["Heterogeneous graphs", "Community detection", "Influence modeling", "Graph sampling (GraphSAGE)"]),
        ("Molecular Graphs", ["Atom/bond features", "Message passing", "Global pooling", "Property prediction / docking"]),
        ("Recommender Systems", ["User-item bipartite", "LightGCN propagation", "Temporal dynamics", "Counterfactual debiasing"]),
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    x_positions = [0.5, 3.7, 6.9]

    for (title, bullets), x, color in zip(domains, x_positions, colors):
        rect = patches.FancyBboxPatch(
            (x, 0.8),
            2.5,
            2.8,
            boxstyle="round,pad=0.3",
            linewidth=1.5,
            edgecolor=color,
            facecolor=(0.94, 0.97, 1.0),
        )
        ax.add_patch(rect)
        ax.text(x + 1.25, 3.3, title, ha="center", va="center", fontsize=12, color=color, fontweight="bold")
        y = 2.9
        for bullet in bullets:
            ax.text(x + 1.25, y, f"• {bullet}", ha="left", va="center", fontsize=10)
            y -= 0.6

    ax.text(
        4.5,
        0.3,
        "Engineering themes: scalability, explainability, robustness",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#9467bd"),
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gnn_application_landscape.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_gcn_graph_structure()
    plot_gcn_layer_flow()
    plot_gnn_application_landscape()


if __name__ == "__main__":
    main()
