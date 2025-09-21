#!/usr/bin/env python3
"""Generate figures for the A2C tutorial."""

import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "a2c_returns.png",
    "a2c_value_baseline.png",
]


@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    start_state: int = 20
    goal_state: int = 4
    step_penalty: float = -0.01
    goal_reward: float = 1.0

    def __post_init__(self) -> None:
        self.n_states = self.width * self.height
        self.actions = [0, 1, 2, 3]  # up, right, down, left

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.width)

    def coord_to_state(self, coord: Tuple[int, int]) -> int:
        return coord[0] * self.width + coord[1]

    def reset(self) -> int:
        return self.start_state

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        if state == self.goal_state:
            return state, 0.0, True

        row, col = self.state_to_coord(state)
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.width - 1, col + 1)
        elif action == 2:
            row = min(self.height - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)

        next_state = self.coord_to_state((row, col))
        reward = self.step_penalty
        done = False
        if next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        return next_state, reward, done


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    exp = np.exp(z)
    return exp / exp.sum()


def run_a2c(
    env: GridWorld,
    episodes: int = 600,
    actor_lr: float = 0.15,
    critic_lr: float = 0.1,
    gamma: float = 0.96,
    entropy_coef: float = 0.01,
    seed: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    n_states = env.n_states
    n_actions = len(env.actions)

    theta = rng.normal(scale=0.05, size=(n_states, n_actions))
    V = np.zeros(n_states, dtype=float)
    returns: List[float] = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        steps = 0

        while not done and steps < 60:
            logits = theta[state]
            probs = softmax(logits)
            action = rng.choice(n_actions, p=probs)

            next_state, reward, done = env.step(state, action)

            td_target = reward + gamma * V[next_state] * (1.0 - float(done))
            td_error = td_target - V[state]

            V[state] += critic_lr * td_error

            grad_log = -probs
            grad_log[action] += 1.0
            theta[state] += actor_lr * (td_error * grad_log + entropy_coef * (-np.log(probs + 1e-8) - 1.0))

            episode_return += reward
            state = next_state
            steps += 1

        returns.append(episode_return)

    return theta, V, returns


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
    ax.set_title("A2C episode returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_value_heatmap(V: np.ndarray, env: GridWorld, output_path: str) -> None:
    heat = V.reshape((env.height, env.width))
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(heat, cmap="viridis")
    for i in range(env.height):
        for j in range(env.width):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Learned state values")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    env = GridWorld()
    theta, V, returns = run_a2c(env)

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))
    plot_value_heatmap(V, env, os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()