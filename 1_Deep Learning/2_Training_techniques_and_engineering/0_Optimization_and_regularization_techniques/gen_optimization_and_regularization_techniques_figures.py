"""
Generate illustrative figures for the optimization and regularization techniques tutorials.

The script produces the following PNG files in the local figures/ directory:
  - optimizer_dynamics.png
  - normalization_comparison.png
  - learning_rate_policies.png
  - augmentation_transfer_pipeline.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def _contour_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Elliptical valley with mild rotation to illustrate optimizer trajectories."""
    a, b, c = 1.0, 0.15, 0.5
    return a * x**2 + b * y**2 + c * x * y


def _gradient(theta: np.ndarray) -> np.ndarray:
    x, y = theta
    grad_x = 2 * x + 0.5 * y
    grad_y = 0.3 * y + 0.5 * x
    return np.array([grad_x, grad_y])


def _simulate_optimizers(steps: int = 80) -> Dict[str, np.ndarray]:
    """Simulate parameter trajectories for several optimizers on the toy valley."""
    start = np.array([2.5, 2.0])

    trajectories: Dict[str, np.ndarray] = {}

    # SGD with fixed learning rate
    lr_sgd = 0.08
    theta = start.copy()
    path = [theta.copy()]
    for _ in range(steps):
        theta = theta - lr_sgd * _gradient(theta)
        path.append(theta.copy())
    trajectories["SGD"] = np.array(path)

    # Momentum SGD
    lr_mom, momentum = 0.08, 0.9
    theta = start.copy()
    velocity = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(steps):
        velocity = momentum * velocity + _gradient(theta)
        theta = theta - lr_mom * velocity
        path.append(theta.copy())
    trajectories["Momentum"] = np.array(path)

    # RMSprop
    lr_rms, decay, eps = 0.06, 0.9, 1e-8
    theta = start.copy()
    sq_avg = np.zeros_like(theta)
    path = [theta.copy()]
    for _ in range(steps):
        grad = _gradient(theta)
        sq_avg = decay * sq_avg + (1 - decay) * grad**2
        theta = theta - lr_rms * grad / (np.sqrt(sq_avg) + eps)
        path.append(theta.copy())
    trajectories["RMSprop"] = np.array(path)

    # Adam
    lr_adam, beta1, beta2 = 0.08, 0.9, 0.999
    theta = start.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    path = [theta.copy()]
    for t in range(1, steps + 1):
        grad = _gradient(theta)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - lr_adam * m_hat / (np.sqrt(v_hat) + eps)
        path.append(theta.copy())
    trajectories["Adam"] = np.array(path)

    return trajectories


def plot_optimizer_dynamics() -> None:
    _configure_matplotlib()
    trajectories = _simulate_optimizers()

    x = np.linspace(-3.0, 3.0, 200)
    y = np.linspace(-3.0, 3.0, 200)
    X, Y = np.meshgrid(x, y)
    Z = _contour_function(X, Y)

    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

    colors = {"SGD": "#d62728", "Momentum": "#9467bd", "RMSprop": "#ff7f0e", "Adam": "#2ca02c"}
    for name, traj in trajectories.items():
        ax.plot(traj[:, 0], traj[:, 1], label=name, color=colors[name])
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[name], marker="o", s=30)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[name], marker="*", s=80)

    ax.set_title("Optimizer Trajectories on a Curved Valley")
    ax.set_xlabel("$\\theta_1$")
    ax.set_ylabel("$\\theta_2$")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "optimizer_dynamics.png", dpi=300)
    plt.close(fig)


def _batch_norm(data: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0, keepdims=True)
    var = data.var(axis=0, keepdims=True)
    normalized = (data - mean) / np.sqrt(var + eps)
    return normalized, mean.squeeze(), var.squeeze()


def _layer_norm(data: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=1, keepdims=True)
    var = data.var(axis=1, keepdims=True)
    normalized = (data - mean) / np.sqrt(var + eps)
    return normalized, mean.squeeze(), var.squeeze()


def plot_normalization_comparison() -> None:
    _configure_matplotlib()
    rng = np.random.default_rng(seed=42)
    batch = rng.normal(loc=[2.0, -1.5, 0.5, 3.0], scale=[2.0, 0.8, 1.5, 0.5], size=(64, 4))
    batch += rng.normal(scale=0.3, size=batch.shape)

    bn_data, bn_mean, bn_var = _batch_norm(batch)
    ln_data, ln_mean, ln_var = _layer_norm(batch)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    feature_ids = np.arange(batch.shape[1])

    axes[0, 0].set_title("Batch Statistics (per feature)")
    axes[0, 0].bar(feature_ids - 0.2, batch.mean(axis=0), width=0.4, label="Mean", color="#1f77b4")
    axes[0, 0].bar(feature_ids + 0.2, batch.var(axis=0), width=0.4, label="Variance", color="#ff7f0e")
    axes[0, 0].set_xticks(feature_ids)
    axes[0, 0].legend()

    axes[0, 1].set_title("BatchNorm Outputs")
    for i in range(4):
        axes[0, 1].hist(bn_data[:, i], bins=15, alpha=0.6, label=f"Feature {i}")
    axes[0, 1].legend()

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.5,
        0.5,
        "\\n".join(
            [
                "BatchNorm:",
                f"$\\mu_\\mathcal{{B}} = {[f'{m:.2f}' for m in bn_mean]}$",
                f"$\\sigma^2_\\mathcal{{B}} = {[f'{v:.2f}' for v in bn_var]}$",
            ]
        ),
        ha="center",
        va="center",
    )

    axes[1, 0].set_title("Layer Statistics (per sample)")
    axes[1, 0].plot(ln_mean, label="Mean")
    axes[1, 0].plot(ln_var, label="Variance")
    axes[1, 0].set_xlabel("Sample index")
    axes[1, 0].legend()

    axes[1, 1].set_title("LayerNorm Outputs (sample 0)")
    axes[1, 1].bar(feature_ids, ln_data[0], color="#2ca02c")
    axes[1, 1].set_xticks(feature_ids)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.5,
        0.5,
        "\\n".join(
            [
                "LayerNorm:",
                "$\\mu = \\frac{1}{d} \\sum_j h_j$",
                "$\\sigma^2 = \\frac{1}{d} \\sum_j (h_j - \\mu)^2$",
                "适用于任意 batch 大小",
            ]
        ),
        ha="center",
        va="center",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "normalization_comparison.png", dpi=300)
    plt.close(fig)


def plot_learning_rate_policies() -> None:
    _configure_matplotlib()
    total_steps = 1000
    warmup_steps = 100
    steps = np.arange(total_steps)

    eta0 = 1e-3
    eta_min = 1e-5
    gamma = 0.6
    k = 200

    step_decay = eta0 * (gamma ** (steps // k))
    exp_decay = eta0 * np.exp(-0.0025 * steps)
    cosine = eta_min + 0.5 * (eta0 - eta_min) * (1 + np.cos(np.pi * (steps - warmup_steps) / (total_steps - warmup_steps)))
    cosine[:warmup_steps] = np.linspace(0, eta0, warmup_steps)

    cycle_length = 250
    cyclical = eta_min + (eta0 - eta_min) * (1 - np.abs((steps % cycle_length) / (cycle_length / 2) - 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, step_decay, label="Step decay")
    ax.plot(steps, exp_decay, label="Exponential decay")
    ax.plot(steps, cosine, label="Cosine w/ warm-up")
    ax.plot(steps, cyclical, label="Triangular CLR")
    ax.axvspan(0, warmup_steps, color="#d3d3d3", alpha=0.3, label="Warm-up window")

    ax.set_title("Learning Rate Scheduling Strategies")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_ylim(0, eta0 * 1.1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_rate_policies.png", dpi=300)
    plt.close(fig)


def plot_augmentation_transfer_pipeline() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    def add_box(x: float, y: float, text: str, width: float = 1.6, height: float = 0.8, color: str = "#1f77b4") -> patches.FancyBboxPatch:
        box = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.1",
            linewidth=1.5,
            edgecolor=color,
            facecolor=(0.9, 0.95, 1.0),
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11, wrap=True)
        return box

    box_specs = [
        (0.2, 1.1, "Raw Dataset\n$\\mathcal{D}_{\\mathrm{src}}$", "#1f77b4"),
        (2.1, 1.1, "Rich Augmentations\nFlip / Color Jitter / Mixup", "#ff7f0e"),
        (4.0, 1.1, "Pretrained Backbone\n(ResNet / ViT / BERT)", "#2ca02c"),
        (5.9, 1.1, "Transfer Strategy\nFreeze / Partial Fine-tune", "#9467bd"),
        (7.8, 1.1, "Target Task Head\nClassifier / Regressor", "#d62728"),
    ]
    boxes = [add_box(x, y, text, color=color) for x, y, text, color in box_specs]

    for (x, y, _, _), (x_next, y_next, _, _) in zip(box_specs, box_specs[1:]):
        start = (x + 1.6, y + 0.4)
        end = (x_next, y_next + 0.4)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#555555"),
        )

    # Augmentation branches
    augment_y_positions = [0.3, 0.0, -0.3]
    augment_labels = ["Geometric", "Photometric", "Mix-based"]
    for offset, label in zip(augment_y_positions, augment_labels):
        ax.annotate(
            label,
            xy=(2.1 + 0.8, 1.1),
            xytext=(2.1 + 0.8, 0.8 + offset),
            arrowprops=dict(arrowstyle="-|>", color="#ff7f0e"),
            ha="center",
        )

    ax.text(
        4.8,
        0.2,
        "Self-supervised objectives\n(Contrastive / Masked modeling)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="#2ca02c"),
    )

    ax.text(
        6.8,
        0.3,
        "Regularization:\nL2-SP / Fisher / Grad clipping",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="#9467bd"),
    )

    ax.set_xlim(0, 9.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title("Data Augmentation & Transfer Learning Pipeline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "augmentation_transfer_pipeline.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_optimizer_dynamics()
    plot_normalization_comparison()
    plot_learning_rate_policies()
    plot_augmentation_transfer_pipeline()


if __name__ == "__main__":
    main()
