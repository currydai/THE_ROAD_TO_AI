#!/usr/bin/env python3
"""Generate figures for the neural network basics tutorials."""
from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def figures_dir() -> Path:
    """Return the output directory for generated figures."""
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_perceptron_boundary(out_dir: Path) -> None:
    rng = np.random.default_rng(42)
    w = np.array([1.2, -0.8])
    b = 0.2
    points = rng.normal(size=(200, 2))
    labels = np.sign(points @ w + b)

    grid_x = np.linspace(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5, 100)
    boundary_y = (-w[0] * grid_x - b) / w[1]

    margin = 1.0 / np.linalg.norm(w)
    margin_y_plus = (-w[0] * grid_x - b + margin) / w[1]
    margin_y_minus = (-w[0] * grid_x - b - margin) / w[1]

    plt.figure(figsize=(6, 5))
    plt.scatter(points[labels > 0, 0], points[labels > 0, 1], c="#1f77b4", label="Positive class", alpha=0.75)
    plt.scatter(points[labels < 0, 0], points[labels < 0, 1], c="#ff7f0e", label="Negative class", alpha=0.75)
    plt.plot(grid_x, boundary_y, "k-", label="Decision boundary")
    plt.plot(grid_x, margin_y_plus, "k--", linewidth=1.0, label="Margin")
    plt.plot(grid_x, margin_y_minus, "k--", linewidth=1.0)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Perceptron decision boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "perceptron_decision_boundary.png", dpi=300)
    plt.close()


def plot_mlp_forward_pass(out_dir: Path) -> None:
    layers = [3, 4, 4, 2]
    spacing_x = 2.2
    spacing_y = 1.2
    fig, ax = plt.subplots(figsize=(7, 4))

    for layer_idx, layer_size in enumerate(layers):
        x = layer_idx * spacing_x
        y_positions = np.linspace(-(layer_size - 1) * spacing_y / 2, (layer_size - 1) * spacing_y / 2, layer_size)
        ax.scatter([x] * layer_size, y_positions, s=600, facecolors="#ffffff", edgecolors="#4c4c4c", zorder=3)
        for node_idx, y in enumerate(y_positions):
            label = f"h$^{({layer_idx})}@{{{node_idx+1}}}$" if layer_idx not in {0, len(layers) - 1} else ""
            if layer_idx == 0:
                label = f"x$_{node_idx+1}$"
            elif layer_idx == len(layers) - 1:
                label = f"y$_{node_idx+1}$"
            if label:
                ax.text(x, y, label, ha="center", va="center", fontsize=10)
        if layer_idx > 0:
            prev_size = layers[layer_idx - 1]
            prev_y = np.linspace(-(prev_size - 1) * spacing_y / 2, (prev_size - 1) * spacing_y / 2, prev_size)
            for i, y_prev in enumerate(prev_y):
                for j, y_curr in enumerate(y_positions):
                    ax.plot([x - spacing_x, x], [y_prev, y_curr], color="#999999", linewidth=0.8, zorder=1)

    ax.axis("off")
    ax.set_title("Forward propagation through an MLP")
    plt.tight_layout()
    fig.savefig(out_dir / "mlp_forward_pass.png", dpi=300)
    plt.close(fig)


def gelu(x: np.ndarray) -> np.ndarray:
     return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def plot_activation_functions(out_dir: Path) -> None:
    x = np.linspace(-5, 5, 1000)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0.0, x)
    leaky_relu = np.where(x >= 0.0, x, 0.1 * x)
    gelu_vals = gelu(x)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, sigmoid, label="Sigmoid", linewidth=2)
    plt.plot(x, tanh, label="Tanh", linewidth=2)
    plt.plot(x, relu, label="ReLU", linewidth=2)
    plt.plot(x, leaky_relu, label="Leaky ReLU", linewidth=2)
    plt.plot(x, gelu_vals, label="GELU", linewidth=2)
    plt.xlabel("Input")
    plt.ylabel("Activation")
    plt.title("Activation function comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "activation_functions.png", dpi=300)
    plt.close()


def plot_loss_functions(out_dir: Path) -> None:
    p = np.linspace(0.001, 0.999, 1000)
    target = 1.0
    residual = p - target

    mse = residual ** 2
    bce = -np.log(p)
    delta = 0.1
    huber = np.where(np.abs(residual) <= delta, 0.5 * residual ** 2, delta * (np.abs(residual) - 0.5 * delta))

    plt.figure(figsize=(7, 4.5))
    plt.plot(p, mse, label="MSE (target=1)", linewidth=2)
    plt.plot(p, bce, label="Binary cross-entropy", linewidth=2)
    plt.plot(p, huber, label=f"Huber (delta={delta})", linewidth=2)
    plt.xlabel("Predicted probability for the positive class")
    plt.ylabel("Loss")
    plt.title("Loss function comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_functions.png", dpi=300)
    plt.close()


def main() -> None:
    out_dir = figures_dir()
    plot_perceptron_boundary(out_dir)
    plot_mlp_forward_pass(out_dir)
    plot_activation_functions(out_dir)
    plot_loss_functions(out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
