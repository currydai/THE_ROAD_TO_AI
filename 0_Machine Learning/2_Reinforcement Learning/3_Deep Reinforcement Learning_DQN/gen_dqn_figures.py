#!/usr/bin/env python3
"""Generate figures for the DQN tutorial."""

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "dqn_returns.png",
    "dqn_q_values.png",
]


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


@dataclass
class SimpleEnv:
    action_magnitude: float = 0.4
    drift: float = 0.9
    noise_std: float = 0.02

    def reset(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(-1.5, 1.5))

    def step(self, state: float, action: int, rng: np.random.Generator) -> Tuple[float, float, bool]:
        act = (action - 0.5) * 2 * self.action_magnitude  # map {0,1} -> {-mag, +mag}
        next_state = self.drift * state + act + rng.normal(scale=self.noise_std)
        reward = - (state ** 2 + 0.1 * act ** 2)
        done = False
        return float(next_state), float(reward), done


def init_network(input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator) -> dict:
    limit1 = np.sqrt(6 / (input_dim + hidden_dim))
    limit2 = np.sqrt(6 / (hidden_dim + output_dim))
    return {
        "W1": rng.uniform(-limit1, limit1, size=(hidden_dim, input_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.uniform(-limit2, limit2, size=(output_dim, hidden_dim)),
        "b2": np.zeros(output_dim),
    }


def forward(network: dict, state_batch: np.ndarray) -> Tuple[np.ndarray, dict]:
    z1 = state_batch @ network["W1"].T + network["b1"]
    h1 = np.tanh(z1)
    q = h1 @ network["W2"].T + network["b2"]
    cache = {"h1": h1}
    return q, cache


def update_network(network: dict, grads: dict, lr: float) -> None:
    for key in network:
        network[key] -= lr * grads[key]


def soft_update(target: dict, source: dict, tau: float) -> None:
    for key in target:
        target[key] = (1 - tau) * target[key] + tau * source[key]


def train_dqn(
    env: SimpleEnv,
    episodes: int = 400,
    horizon: int = 40,
    gamma: float = 0.95,
    lr: float = 0.01,
    tau: float = 0.01,
    buffer_size: int = 5000,
    batch_size: int = 64,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.05,
    seed: int = 0,
) -> Tuple[dict, List[float]]:
    rng = np.random.default_rng(seed)
    replay: Deque[Tuple[float, int, float, float]] = deque(maxlen=buffer_size)

    online = init_network(1, 32, 2, rng)
    target = init_network(1, 32, 2, rng)
    soft_update(target, online, tau=1.0)

    returns: List[float] = []

    for episode in range(episodes):
        state = env.reset(rng)
        episode_return = 0.0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / (episodes / 3))

        for _ in range(horizon):
            if rng.random() < epsilon:
                action = rng.integers(2)
            else:
                q_values, _ = forward(online, np.array([[state]]))
                action = int(np.argmax(q_values[0]))

            next_state, reward, done = env.step(state, action, rng)
            replay.append((state, action, reward, next_state))

            state = next_state
            episode_return += reward

            if len(replay) >= batch_size:
                batch_idx = rng.choice(len(replay), size=batch_size, replace=False)
                batch = [replay[i] for i in batch_idx]
                states = np.array([[item[0]] for item in batch])
                actions = np.array([item[1] for item in batch])
                rewards = np.array([item[2] for item in batch])
                next_states = np.array([[item[3]] for item in batch])

                q_next, _ = forward(target, next_states)
                targets = rewards + gamma * np.max(q_next, axis=1)

                q_values, cache = forward(online, states)
                y = q_values.copy()
                y[np.arange(batch_size), actions] = targets

                # backprop through tanh hidden layer
                error = (q_values - y) / batch_size
                grads = {
                    "W2": error.T @ cache["h1"],
                    "b2": error.sum(axis=0),
                }
                dh = error @ online["W2"] * (1 - cache["h1"] ** 2)
                grads["W1"] = dh.T @ states
                grads["b1"] = dh.sum(axis=0)

                update_network(online, grads, lr)
                soft_update(target, online, tau)

        returns.append(episode_return)

    return online, returns


def evaluate_q_values(network: dict, states: np.ndarray) -> np.ndarray:
    q_values, _ = forward(network, states)
    return q_values


def plot_returns(returns: List[float], output_path: str) -> None:
    window = 20
    if len(returns) >= window:
        running = np.convolve(returns, np.ones(window) / window, mode="valid")
    else:
        running = np.array(returns)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(returns, alpha=0.35, label="Episode return")
    ax.plot(np.arange(len(running)) + window - 1, running, color="#ff7f0e", linewidth=1.8, label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("DQN episode returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_q_heatmap(q_values: np.ndarray, states: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    im = ax.imshow(q_values.T, aspect="auto", cmap="viridis", extent=(states.min(), states.max(), -0.5, 1.5))
    ax.set_xlabel("State")
    ax.set_ylabel("Action")
    ax.set_title("State-action values (Q)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["left", "right"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    env = SimpleEnv()
    network, returns = train_dqn(env)

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))

    state_grid = np.linspace(-2.0, 2.0, num=80)[:, None]
    q_values = evaluate_q_values(network, state_grid)
    plot_q_heatmap(q_values, state_grid.squeeze(), os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()