#!/usr/bin/env python3
"""Generate illustrative figures for the training deep networks tutorial."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_figures_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_backprop_graph(out_dir: Path) -> None:
    layers = [3, 4, 4, 2]
    spacing_x = 2.2
    spacing_y = 1.2
    fig, ax = plt.subplots(figsize=(7, 4))

    forward_color = "#1f77b4"
    backward_color = "#d62728"

    node_positions = {}
    for layer_idx, layer_size in enumerate(layers):
        x = layer_idx * spacing_x
        y_positions = np.linspace(-(layer_size - 1) * spacing_y / 2, (layer_size - 1) * spacing_y / 2, layer_size)
        node_positions[layer_idx] = [(x, y) for y in y_positions]
        ax.scatter([x] * layer_size, y_positions, s=600, facecolors="#ffffff", edgecolors="#4c4c4c", zorder=3)
        for node_idx, y in enumerate(y_positions):
            if layer_idx == 0:
                label = f"x$_{node_idx+1}$"
            elif layer_idx == len(layers) - 1:
                label = f"y$_{node_idx+1}$"
            else:
                label = f"h$_{layer_idx,{node_idx+1}}$"
            ax.text(x, y, label, ha="center", va="center", fontsize=9)

    for layer_idx in range(len(layers) - 1):
        for (x0, y0) in node_positions[layer_idx]:
            for (x1, y1) in node_positions[layer_idx + 1]:
                ax.annotate(
                    "",
                    xy=(x1 - 0.25, y1),
                    xytext=(x0 + 0.25, y0),
                    arrowprops=dict(arrowstyle="->", color=forward_color, linewidth=0.7),
                    zorder=1,
                )
                ax.annotate(
                    "",
                    xy=(x0 + 0.2, y0 + 0.05),
                    xytext=(x1 - 0.2, y1 + 0.05),
                    arrowprops=dict(arrowstyle="->", color=backward_color, linewidth=0.7, ls="--"),
                    zorder=2,
                )

    ax.text(spacing_x * 1.2, spacing_y * 2.2, "Forward", color=forward_color, fontsize=11)
    ax.text(spacing_x * 1.2, spacing_y * 1.8, "Backward", color=backward_color, fontsize=11)
    ax.axis("off")
    ax.set_title("Forward and backward signals in backprop")
    fig.tight_layout()
    fig.savefig(out_dir / "backprop_computational_graph.png", dpi=300)
    plt.close(fig)


def quadratic_function(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([[3.0, 0.8], [0.8, 1.0]])
    grad = A @ x
    return 0.5 * x.T @ A @ x, grad


def simulate_optimizer(method: str, steps: int = 50, eta: float = 0.4, mu: float = 0.9) -> np.ndarray:
    x = np.array([2.5, -1.5])
    trajectory = [x.copy()]
    v = np.zeros_like(x)
    for t in range(steps):
        _, grad = quadratic_function(x if method != "nag" else x + mu * v)
        if method == "sgd":
            x = x - eta * grad
        elif method == "momentum":
            v = mu * v - eta * grad
            x = x + v
        elif method == "nag":
            v = mu * v - eta * grad
            x = x + v
        else:
            raise ValueError(method)
        trajectory.append(x.copy())
    return np.array(trajectory)


def plot_optimization_trajectories(out_dir: Path) -> None:
    traj_sgd = simulate_optimizer("sgd", eta=0.2)
    traj_momentum = simulate_optimizer("momentum", eta=0.25, mu=0.85)
    traj_nag = simulate_optimizer("nag", eta=0.28, mu=0.9)

    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    z = 0.5 * (3 * grid_x**2 + 2 * grid_x * grid_y + grid_y**2)

    fig, ax = plt.subplots(figsize=(6, 5))
    contours = ax.contour(grid_x, grid_y, z, levels=20, cmap="Greys", alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    ax.plot(traj_sgd[:, 0], traj_sgd[:, 1], "o-", label="SGD", color="#1f77b4", markersize=3)
    ax.plot(traj_momentum[:, 0], traj_momentum[:, 1], "o-", label="Momentum", color="#ff7f0e", markersize=3)
    ax.plot(traj_nag[:, 0], traj_nag[:, 1], "o-", label="Nesterov", color="#2ca02c", markersize=3)
    ax.scatter(0, 0, c="black", marker="*", s=120, label="Optimum")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Optimization trajectories on a quadratic bowl")
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    fig.tight_layout()
    fig.savefig(out_dir / "optimization_trajectories.png", dpi=300)
    plt.close(fig)


def plot_learning_rate_schedules(out_dir: Path) -> None:
    steps = np.arange(0, 100)
    warmup = 10
    eta_max = 0.1
    eta_min = 0.0005

    step_decay = np.where(steps < warmup, eta_max * steps / warmup, eta_max * 0.5 ** ((steps - warmup) // 20))
    exp_decay = np.where(steps < warmup, eta_max * steps / warmup, eta_max * np.exp(-0.03 * (steps - warmup)))
    cosine = np.where(
        steps < warmup,
        eta_max * steps / warmup,
        eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (steps - warmup) / (steps.max() - warmup)))
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, step_decay, label="Step decay", linewidth=2)
    ax.plot(steps, exp_decay, label="Exponential", linewidth=2)
    ax.plot(steps, cosine, label="Cosine annealing", linewidth=2)
    ax.axvspan(0, warmup, color="#d3d3d3", alpha=0.4, label="Warmup")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedules")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_rate_schedules.png", dpi=300)
    plt.close(fig)


def plot_regularization_effects(out_dir: Path) -> None:
    epochs = np.arange(0, 100)
    baseline_train = 0.55 + 0.35 * (1 - np.exp(-epochs / 20))
    baseline_val = 0.52 + 0.25 * (1 - np.exp(-epochs / 25)) - 0.05 * (epochs / epochs.max())
    regularized_train = 0.5 + 0.4 * (1 - np.exp(-epochs / 22))
    regularized_val = 0.48 + 0.35 * (1 - np.exp(-epochs / 28))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, baseline_train, label="Train (baseline)", color="#1f77b4", linestyle="--", linewidth=2)
    ax.plot(epochs, baseline_val, label="Val (baseline)", color="#1f77b4", linewidth=2)
    ax.plot(epochs, regularized_train, label="Train (regularized)", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.plot(epochs, regularized_val, label="Val (regularized)", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Regularization reduces generalization gap")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "regularization_effects.png", dpi=300)
    plt.close(fig)


def main() -> None:
    out_dir = ensure_figures_dir()
    plot_backprop_graph(out_dir)
    plot_optimization_trajectories(out_dir)
    plot_learning_rate_schedules(out_dir)
    plot_regularization_effects(out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
