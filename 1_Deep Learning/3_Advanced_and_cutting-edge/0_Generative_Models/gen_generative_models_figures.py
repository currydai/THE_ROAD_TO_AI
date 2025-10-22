"""
Generate figures for the generative models tutorials.

Outputs saved to figures/:
  - autoencoder_architecture.png
  - vae_latent_space.png
  - gan_training_dynamics.png
  - diffusion_process.png
"""

from __future__ import annotations

from pathlib import Path

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


def plot_autoencoder_architecture() -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    layers = [
        ("Input\n784", 0.5),
        ("Encoder\n512", 2.2),
        ("Encoder\n128", 3.9),
        ("Latent\n32", 5.6),
        ("Decoder\n128", 7.3),
        ("Decoder\n512", 9.0),
        ("Reconstruction\n784", 10.7),
    ]

    colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#9467bd", "#2ca02c", "#2ca02c", "#ff7f0e"]
    for (label, x), color in zip(layers, colors):
        rect = patches.FancyBboxPatch(
            (x, 1.2),
            1.2,
            1.5,
            boxstyle="round,pad=0.2",
            linewidth=1.5,
            edgecolor=color,
            facecolor=(0.95, 0.97, 1.0),
        )
        ax.add_patch(rect)
        ax.text(x + 0.6, 2.0, label, ha="center", va="center", fontsize=11, color=color)

    for (label, x_start), (_, x_end) in zip(layers[:-1], layers[1:]):
        ax.annotate(
            "",
            xy=(x_end, 2.0),
            xytext=(x_start + 1.2, 2.0),
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#555555"),
        )

    ax.text(
        5.6,
        0.6,
        "Latent bottleneck controls compression\nRegularizers enforce structure",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#9467bd"),
    )

    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 3.5)
    ax.set_title("Autoencoder Encoder-Decoder Architecture")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "autoencoder_architecture.png", dpi=300)
    plt.close(fig)


def plot_vae_latent_space() -> None:
    _configure_matplotlib()
    rng = np.random.default_rng(0)
    clusters = [
        (np.array([0.0, 0.0]), np.eye(2) * 0.12),
        (np.array([1.5, 1.2]), np.array([[0.2, 0.08], [0.08, 0.15]])),
        (np.array([-1.2, 1.6]), np.array([[0.15, -0.05], [-0.05, 0.18]])),
        (np.array([-1.6, -1.0]), np.array([[0.1, 0.02], [0.02, 0.14]])),
    ]
    samples = []
    labels = []
    for idx, (mean, cov) in enumerate(clusters):
        s = rng.multivariate_normal(mean, cov, size=200)
        samples.append(s)
        labels.append(np.full(len(s), idx))
    samples = np.vstack(samples)
    labels = np.concatenate(labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(samples[:, 0], samples[:, 1], c=labels, cmap="tab10", alpha=0.7, edgecolors="none")
    ax.scatter(0, 0, c="black", marker="x", s=80, label="Prior origin")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("VAE Latent Space Samples")
    legend = ax.legend(loc="upper right")
    legend.get_texts()[0].set_text("Prior origin")
    ax.annotate(
        "High-density region\nfrom training data",
        xy=(1.6, 1.3),
        xytext=(0.6, 2.2),
        arrowprops=dict(arrowstyle="->", color="#333333"),
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vae_latent_space.png", dpi=300)
    plt.close(fig)


def plot_gan_training_dynamics() -> None:
    _configure_matplotlib()
    steps = np.linspace(0, 100, 500)
    gen_loss = 1.5 * np.exp(-steps / 40) + 0.4 * np.sin(steps / 7) + 0.6
    disc_loss = 1.2 * np.exp(-steps / 60) - 0.3 * np.sin(steps / 5 + 0.4) + 0.5
    fid = 60 * np.exp(-steps / 30) + 5 * np.sin(steps / 8) + 10

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(steps, gen_loss, label="Generator loss", color="#1f77b4")
    ax1.plot(steps, disc_loss, label="Discriminator loss", color="#ff7f0e")
    ax2.plot(steps, fid, label="FID (lower is better)", color="#2ca02c", linestyle="--")

    ax1.set_xlabel("Training steps (k)")
    ax1.set_ylabel("Adversarial losses")
    ax2.set_ylabel("FID")
    ax1.set_title("GAN Training Dynamics with Gradient Penalty and TTUR")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    ax1.axvline(20, color="#9467bd", linestyle=":", linewidth=1.2)
    ax1.text(21, ax1.get_ylim()[1] - 0.3, "Gradient penalty on", color="#9467bd")
    ax1.axvline(60, color="#8c564b", linestyle=":", linewidth=1.2)
    ax1.text(61, ax1.get_ylim()[1] - 0.6, "TTUR adjustment", color="#8c564b")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gan_training_dynamics.png", dpi=300)
    plt.close(fig)


def plot_diffusion_process() -> None:
    _configure_matplotlib()
    timesteps = np.arange(0, 1000)
    cosine_schedule = np.cos(((timesteps / 999) + 0.008) / 1.008 * np.pi / 2) ** 2
    signal_to_noise = cosine_schedule / (1 - cosine_schedule + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(timesteps, cosine_schedule, color="#1f77b4")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("$\\bar{\\alpha}_t$")
    axes[0].set_title("Cumulative Signal ($\\bar{\\alpha}_t$)")

    axes[1].plot(timesteps, 10 * np.log10(signal_to_noise), color="#ff7f0e")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_title("Signal-to-Noise Ratio")

    axes[1].axvline(200, color="#2ca02c", linestyle="--")
    axes[1].text(210, 10, "Early guidance\n(high SNR)", color="#2ca02c")
    axes[1].axvline(800, color="#9467bd", linestyle="--")
    axes[1].text(810, -30, "Late diffusion\n(low SNR)", color="#9467bd")

    fig.suptitle("Forward/Reverse Diffusion Scheduling (Cosine)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "diffusion_process.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_autoencoder_architecture()
    plot_vae_latent_space()
    plot_gan_training_dynamics()
    plot_diffusion_process()


if __name__ == "__main__":
    main()
