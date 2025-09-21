#!/usr/bin/env python3
"""Generate figures for the REINFORCE tutorial."""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "reinforce_returns.png",
    "reinforce_state_visitation.png",
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


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    G = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def run_reinforce(
    grid_width: int = 5,
    grid_height: int = 5,
    episodes: int = 700,
    alpha: float = 0.08,
    baseline_lr: float = 0.05,
    gamma: float = 0.96,
    seed: int = 17,
) -> Tuple[np.ndarray, List[float], np.ndarray]:
    rng = np.random.default_rng(seed)
    n_states = grid_width * grid_height
    n_actions = 4  # up, right, down, left

    theta = rng.normal(scale=0.1, size=(n_states, n_actions))
    baseline = np.zeros(n_states)
    visitation = np.zeros(n_states)

    start_state = n_states - grid_width
    goal_state = grid_width - 1
    obstacle_states = set()

    returns_history: List[float] = []

    def state_to_coord(state: int) -> Tuple[int, int]:
        return divmod(state, grid_width)

    def coord_to_state(coord: Tuple[int, int]) -> int:
        return coord[0] * grid_width + coord[1]

    def step(state: int, action: int) -> Tuple[int, float, bool]:
        row, col = state_to_coord(state)
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(grid_width - 1, col + 1)
        elif action == 2:
            row = min(grid_height - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)
        next_state = coord_to_state((row, col))
        reward = -0.01
        done = False
        if next_state == goal_state:
            reward = 1.0
            done = True
        elif next_state in obstacle_states:
            reward = -0.3
        return next_state, reward, done

    for episode in range(episodes):
        state = start_state
        trajectory: List[Tuple[int, int]] = []
        rewards: List[float] = []
        done = False
        steps = 0

        while not done and steps < 40:
            probs = softmax(theta[state])
            action = rng.choice(n_actions, p=probs)
            next_state, reward, done = step(state, action)

            trajectory.append((state, action))
            rewards.append(reward)
            visitation[state] += 1

            state = next_state
            steps += 1

        returns_episode = compute_returns(rewards, gamma)
        returns_history.append(sum(rewards))

        for (s, a), G_t in zip(trajectory, returns_episode):
            probs = softmax(theta[s])
            grad_log = -probs
            grad_log[a] += 1.0

            baseline[s] += baseline_lr * (G_t - baseline[s])
            advantage = G_t - baseline[s]
            theta[s] += alpha * grad_log * advantage

    return theta, returns_history, visitation


def plot_returns(returns: List[float], output_path: str) -> None:
    window = 30
    if len(returns) >= window:
        running = np.convolve(returns, np.ones(window) / window, mode="valid")
    else:
        running = np.array(returns)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(returns, alpha=0.35, label="Episode return")
    ax.plot(np.arange(len(running)) + window - 1, running, color="#ff7f0e", linewidth=1.8, label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("REINFORCE episode returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_state_visitation(visitation: np.ndarray, grid_width: int, grid_height: int, output_path: str) -> None:
    heat = visitation.reshape((grid_height, grid_width))
    if heat.max() > 0:
        heat = heat / heat.max()
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.imshow(heat, cmap="magma")
    for i in range(grid_height):
        for j in range(grid_width):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Normalized state visitation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    theta, returns, visitation = run_reinforce()

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))
    plot_state_visitation(visitation, 5, 5, os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()