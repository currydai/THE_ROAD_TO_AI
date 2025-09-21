#!/usr/bin/env python3
"""Generate figures for the DDPG tutorial."""

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "ddpg_returns.png",
    "ddpg_policy_trace.png",
]


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


@dataclass
class SimpleContinuousEnv:
    dt: float = 0.1
    drift: float = 0.9
    action_gain: float = 0.3
    noise_std: float = 0.02

    def reset(self, rng: np.random.Generator) -> float:
        return rng.uniform(-1.5, 1.5)

    def step(self, state: float, action: float, rng: np.random.Generator) -> Tuple[float, float, bool]:
        next_state = self.drift * state + self.action_gain * action + rng.normal(scale=self.noise_std)
        reward = - (state ** 2 + 0.1 * action ** 2)
        done = False
        return next_state, reward, done


def actor(state: float, theta: np.ndarray) -> float:
    # Deterministic linear policy mu(s) = theta0 * s + theta1
    return float(theta[0] * state + theta[1])


def actor_grad(state: float) -> np.ndarray:
    return np.array([state, 1.0])


def features(state: float, action: float) -> np.ndarray:
    return np.array([state, action, state * action, state ** 2, action ** 2, 1.0])


def critic_value(state: float, action: float, weights: np.ndarray) -> float:
    return float(weights @ features(state, action))


def critic_grad_action(state: float, action: float, weights: np.ndarray) -> float:
    # derivative of critic wrt action
    phi = features(state, action)
    # partial derivatives: d/d a of features
    dphi_da = np.array([0.0, 1.0, state, 0.0, 2.0 * action, 0.0])
    return float(weights @ dphi_da)


def soft_update(target: np.ndarray, source: np.ndarray, tau: float) -> None:
    target *= (1.0 - tau)
    target += tau * source


def run_ddpg(
    env: SimpleContinuousEnv,
    episodes: int = 400,
    horizon: int = 40,
    gamma: float = 0.95,
    actor_lr: float = 0.01,
    critic_lr: float = 0.02,
    tau: float = 0.02,
    batch_size: int = 64,
    buffer_size: int = 5000,
    noise_std: float = 0.2,
    seed: int = 9,
) -> Tuple[np.ndarray, List[float], List[Tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    buffer: Deque[Tuple[float, float, float, float]] = deque(maxlen=buffer_size)

    theta = rng.normal(scale=0.1, size=2)
    theta_target = theta.copy()
    critic_w = rng.normal(scale=0.1, size=6)
    critic_target_w = critic_w.copy()

    returns: List[float] = []

    for episode in range(episodes):
        state = env.reset(rng)
        episode_return = 0.0
        for t in range(horizon):
            mu = actor(state, theta)
            action = mu + rng.normal(scale=noise_std)
            next_state, reward, done = env.step(state, action, rng)
            buffer.append((state, action, reward, next_state))

            if len(buffer) >= batch_size:
                batch = rng.choice(len(buffer), size=batch_size, replace=False)
                states = np.array([buffer[idx][0] for idx in batch])
                actions = np.array([buffer[idx][1] for idx in batch])
                rewards = np.array([buffer[idx][2] for idx in batch])
                next_states = np.array([buffer[idx][3] for idx in batch])

                next_actions = np.array([actor(s, theta_target) for s in next_states])
                targets = rewards + gamma * np.array([
                    critic_value(s, a, critic_target_w) for s, a in zip(next_states, next_actions)
                ])

                phi = np.array([features(s, a) for s, a in zip(states, actions)])
                critic_w += critic_lr * phi.T @ (targets - phi @ critic_w) / batch_size

                actions_pred = np.array([actor(s, theta) for s in states])
                grad_q = np.array([critic_grad_action(s, a, critic_w) for s, a in zip(states, actions_pred)])
                grad_mu = np.array([actor_grad(s) for s in states])
                actor_gradient = (grad_mu.T * grad_q).mean(axis=1)
                theta += actor_lr * actor_gradient

                soft_update(theta_target, theta, tau)
                soft_update(critic_target_w, critic_w, tau)

            state = next_state
            episode_return += reward
            if done:
                break
        returns.append(episode_return)

    policy_trace = [(s, actor(s, theta)) for s in np.linspace(-2.0, 2.0, num=50)]
    return theta, returns, policy_trace


def plot_returns(returns: List[float], output_path: str) -> None:
    window = 20
    if len(returns) >= window:
        running = np.convolve(returns, np.ones(window) / window, mode="valid")
    else:
        running = np.array(returns)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(returns, alpha=0.35, label="Episode return")
    ax.plot(np.arange(len(running)) + window - 1, running, color="#1f77b4", linewidth=1.8, label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("DDPG episode returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_policy(policy_trace: List[Tuple[float, float]], output_path: str) -> None:
    states = np.array([p[0] for p in policy_trace])
    actions = np.array([p[1] for p in policy_trace])
    optimal = -0.8 * states

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(states, actions, label="Learned policy", linewidth=2.0)
    ax.plot(states, optimal, linestyle="--", color="#d62728", label="Reference optimal")
    ax.set_xlabel("State")
    ax.set_ylabel("Action")
    ax.set_title("Deterministic policy after DDPG training")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    env = SimpleContinuousEnv()
    theta, returns, policy_trace = run_ddpg(env)

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))
    plot_policy(policy_trace, os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()