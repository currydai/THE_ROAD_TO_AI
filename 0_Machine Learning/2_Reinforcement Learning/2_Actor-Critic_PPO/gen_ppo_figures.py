#!/usr/bin/env python3
"""Generate figures for the PPO tutorial."""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "ppo_returns.png",
    "ppo_clip_stats.png",
]


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    start_state: int = 20
    goal_state: int = 4
    step_penalty: float = -0.02
    goal_reward: float = 1.0

    def __post_init__(self) -> None:
        self.n_states = self.width * self.height
        self.n_actions = 4  # up, right, down, left

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


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    exp = np.exp(z)
    return exp / exp.sum()


def collect_trajectories(
    env: GridWorld,
    theta: np.ndarray,
    V: np.ndarray,
    horizon: int,
    batch_size: int,
    gamma: float,
    lam: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    states: List[int] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    log_probs: List[float] = []
    values: List[float] = []
    returns: List[float] = []
    advantages: List[float] = []

    while len(states) < batch_size:
        state = env.reset()
        for _ in range(horizon):
            logits = theta[state]
            probs = softmax(logits)
            action = rng.choice(env.n_actions, p=probs)
            log_prob = np.log(probs[action] + 1e-8)
            value = V[state]

            next_state, reward, done = env.step(state, action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state
            if done or len(states) >= batch_size:
                break

    values.append(V[state])  # bootstrap

    adv = 0.0
    for t in reversed(range(len(states))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        adv = delta + gamma * lam * mask * adv
        advantages.insert(0, adv)
        returns.insert(0, adv + values[t])

    return {
        "states": np.array(states, dtype=int),
        "actions": np.array(actions, dtype=int),
        "returns": np.array(returns, dtype=float),
        "advantages": np.array(advantages, dtype=float),
        "log_probs": np.array(log_probs, dtype=float),
    }


def run_ppo(
    env: GridWorld,
    episodes: int = 400,
    horizon: int = 40,
    gamma: float = 0.96,
    lam: float = 0.9,
    eps_clip: float = 0.2,
    learning_rate: float = 0.08,
    value_lr: float = 0.1,
    entropy_coef: float = 0.01,
    updates_per_batch: int = 4,
    batch_size: int = 256,
    seed: int = 21,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.05, size=(env.n_states, env.n_actions))
    V = np.zeros(env.n_states)

    returns_history: List[float] = []
    clip_fractions: List[float] = []

    for episode in range(episodes):
        data = collect_trajectories(env, theta, V, horizon, batch_size, gamma, lam, rng)
        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns_history.append(np.sum(data["returns"]) / batch_size)

        for _ in range(updates_per_batch):
            idx = rng.choice(batch_size, size=batch_size, replace=False)
            states = data["states"][idx]
            actions = data["actions"][idx]
            old_log_probs = data["log_probs"][idx]
            adv_batch = advantages[idx]
            ret_batch = data["returns"][idx]

            logits = theta[states]
            probs = np.apply_along_axis(softmax, 1, logits)
            new_log_probs = np.log(probs[np.arange(len(states)), actions] + 1e-8)

            ratios = np.exp(new_log_probs - old_log_probs)
            clipped = np.clip(ratios, 1 - eps_clip, 1 + eps_clip)
            policy_loss = -np.mean(np.minimum(ratios * adv_batch, clipped * adv_batch))
            clip_fraction = np.mean(np.abs(ratios - 1.0) > eps_clip / 2)

            entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))

            theta_grad = np.zeros_like(theta)
            for i, s in enumerate(states):
                prob = probs[i]
                one_hot = np.zeros(env.n_actions)
                one_hot[actions[i]] = 1.0
                grad_log = one_hot - prob
                weight = np.where(ratios[i] * adv_batch[i] < clipped[i] * adv_batch[i], ratios[i], clipped[i])
                theta_grad[s] += weight * adv_batch[i] * grad_log
            theta += learning_rate * (theta_grad / len(states)) + entropy_coef * rng.normal(scale=0.01, size=theta.shape)


            value_errors = ret_batch - V[states]
            V[states] += value_lr * value_errors

            clip_fractions.append(float(clip_fraction))

    return theta, V, returns_history, clip_fractions


def plot_returns(returns: List[float], output_path: str) -> None:
    window = 10
    if len(returns) >= window:
        running = np.convolve(returns, np.ones(window) / window, mode="valid")
    else:
        running = np.array(returns)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(returns, alpha=0.4, label="Average return per batch")
    ax.plot(np.arange(len(running)) + window - 1, running, color="#9467bd", linewidth=1.8, label="Moving average")
    ax.set_xlabel("Update iteration")
    ax.set_ylabel("Average return")
    ax.set_title("PPO training returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_clip_stats(clip_fracs: List[float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(clip_fracs, color="#d62728", linewidth=1.4)
    ax.set_xlabel("Policy update step")
    ax.set_ylabel("Clip fraction")
    ax.set_title("PPO clip fraction per update")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    env = GridWorld()
    theta, V, returns, clip_fracs = run_ppo(env)

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))
    plot_clip_stats(clip_fracs, os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()