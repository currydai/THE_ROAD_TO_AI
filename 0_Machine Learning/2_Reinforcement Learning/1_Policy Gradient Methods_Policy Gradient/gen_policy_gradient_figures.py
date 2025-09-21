#!/usr/bin/env python3
"""Generate figures for the policy gradient tutorial."""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "policy_gradient_returns.png",
    "policy_gradient_policy_heatmap.png",
]


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    exp = np.exp(z)
    return exp / exp.sum()


def run_policy_gradient(
    n_states: int = 4,
    n_actions: int = 3,
    episodes: int = 600,
    alpha: float = 0.1,
    baseline_lr: float = 0.05,
    seed: int = 3,
) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.1, size=(n_states, n_actions))
    baseline = np.zeros(n_states)

    # reward mapping: each state has preferred action
    preferred_actions = np.array([0, 2, 1, 0])
    reward_table = np.full((n_states, n_actions), -0.2)
    for s in range(n_states):
        reward_table[s, preferred_actions[s]] = 1.0

    returns: List[float] = []

    for episode in range(episodes):
        state = rng.integers(n_states)
        episode_rewards: List[float] = []
        transitions: List[Tuple[int, int]] = []

        for _ in range(5):
            logits = theta[state]
            policy = softmax(logits)
            action = rng.choice(n_actions, p=policy)
            reward = reward_table[state, action] + rng.normal(scale=0.05)
            next_state = rng.choice(n_states, p=np.full(n_states, 1 / n_states))

            transitions.append((state, action))
            episode_rewards.append(reward)

            state = next_state

        # compute return with discount gamma=0.95
        G = 0.0
        gamma = 0.95
        returns_episode = []
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns_episode.insert(0, G)
        returns.append(sum(episode_rewards))

        for (s, a), discounted_return in zip(transitions, returns_episode):
            logits = theta[s]
            policy = softmax(logits)
            grad_log = -policy
            grad_log[a] += 1.0

            baseline[s] += baseline_lr * (discounted_return - baseline[s])
            advantage = discounted_return - baseline[s]
            theta[s] += alpha * grad_log * advantage

    return theta, returns


def plot_returns(returns: List[float], output_path: str) -> None:
    window = 20
    if len(returns) >= window:
        running = np.convolve(returns, np.ones(window) / window, mode="valid")
    else:
        running = np.array(returns)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(returns, alpha=0.4, label="Episode return")
    ax.plot(np.arange(len(running)) + window - 1, running, color="#d62728", linewidth=1.8, label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Policy gradient episode returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_policy_heatmap(theta: np.ndarray, output_path: str) -> None:
    probs = np.apply_along_axis(softmax, 1, theta)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(probs, cmap="viridis")
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title("Softmax policy after training")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    theta, returns = run_policy_gradient()

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))
    plot_policy_heatmap(theta, os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()