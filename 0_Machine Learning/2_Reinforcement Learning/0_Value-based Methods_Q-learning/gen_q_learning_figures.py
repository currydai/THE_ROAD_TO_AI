#!/usr/bin/env python3
"""Generate figures for the Q-learning tutorial."""

import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "q_learning_returns.png",
    "q_learning_state_values.png",
]


@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    terminal_states: Tuple[int, ...] = (24,)
    obstacle_states: Tuple[int, ...] = ()
    goal_reward: float = 1.0
    step_penalty: float = -0.01

    def __post_init__(self) -> None:
        self.n_states = self.width * self.height
        self.actions = [0, 1, 2, 3]  # up, right, down, left

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.width)

    def coord_to_state(self, coord: Tuple[int, int]) -> int:
        return coord[0] * self.width + coord[1]

    def reset(self) -> int:
        start = 0
        return start

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        if state in self.terminal_states:
            return state, 0.0, True

        row, col = self.state_to_coord(state)
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.width - 1, col + 1)
        elif action == 2:  # down
            row = min(self.height - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)

        next_state = self.coord_to_state((row, col))
        reward = self.step_penalty
        done = False
        if next_state in self.terminal_states:
            reward = self.goal_reward
            done = True
        if next_state in self.obstacle_states:
            reward = -0.2
        return next_state, reward, done


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return rng.integers(len(q_values))
    return int(np.argmax(q_values))


def q_learning(
    env: GridWorld,
    num_episodes: int = 500,
    alpha: float = 0.2,
    gamma: float = 0.97,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.05,
) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(0)
    Q = np.zeros((env.n_states, len(env.actions)), dtype=float)
    returns: List[float] = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / (num_episodes / 3))

        while not done:
            action = epsilon_greedy(Q[state], epsilon, rng)
            next_state, reward, done = env.step(state, action)
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next * (1.0 - float(done))
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            episode_return += reward

        returns.append(episode_return)

    return Q, returns


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
    ax.set_title("Q-learning episode returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_state_values(Q: np.ndarray, env: GridWorld, output_path: str) -> None:
    V = np.max(Q, axis=1).reshape((env.height, env.width))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(V, cmap="viridis")
    for i in range(env.height):
        for j in range(env.width):
            ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Greedy state values after training")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    env = GridWorld()
    Q, returns = q_learning(env)

    returns_path = os.path.join(figures_dir, FIGURE_LIST[0])
    plot_returns(returns, returns_path)

    values_path = os.path.join(figures_dir, FIGURE_LIST[1])
    plot_state_values(Q, env, values_path)

    for figure in FIGURE_LIST:
        print(os.path.join(figures_dir, figure))


if __name__ == "__main__":
    main()