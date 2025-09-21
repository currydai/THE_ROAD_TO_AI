#!/usr/bin/env python3
"""Generate figures for the SAC tutorial."""

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "sac_returns.png",
    "sac_temperature.png",
]


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


@dataclass
class SimpleContinuousEnv:
    drift: float = 0.9
    action_gain: float = 0.35
    noise_std: float = 0.02

    def reset(self, rng: np.random.Generator) -> float:
        return rng.uniform(-1.5, 1.5)

    def step(self, state: float, action: float, rng: np.random.Generator) -> Tuple[float, float, bool]:
        next_state = self.drift * state + self.action_gain * action + rng.normal(scale=self.noise_std)
        reward = - (state ** 2 + 0.15 * action ** 2)
        done = False
        return next_state, reward, done


def features(state: float, action: float) -> np.ndarray:
    return np.array([state, action, state * action, state ** 2, action ** 2, 1.0])


def critic_value(state: float, action: float, weights: np.ndarray) -> float:
    return float(weights @ features(state, action))


def critic_grad_action(state: float, action: float, weights: np.ndarray) -> float:
    dphi_da = np.array([0.0, 1.0, state, 0.0, 2.0 * action, 0.0])
    return float(weights @ dphi_da)


def soft_update(target: np.ndarray, source: np.ndarray, tau: float) -> None:
    target *= (1.0 - tau)
    target += tau * source


def actor_forward(
    state: float,
    theta_mean: np.ndarray,
    log_std: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float]:
    mean = theta_mean[0] * state + theta_mean[1]
    std = np.exp(log_std)
    eps = rng.normal()
    action = mean + std * eps
    log_prob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return action, mean, std, log_prob, eps


def run_sac(
    env: SimpleContinuousEnv,
    episodes: int = 400,
    horizon: int = 40,
    gamma: float = 0.95,
    tau: float = 0.02,
    actor_lr: float = 0.01,
    critic_lr: float = 0.03,
    alpha_lr: float = 0.005,
    target_entropy: float = -1.0,
    batch_size: int = 64,
    buffer_size: int = 6000,
    seed: int = 31,
) -> Tuple[np.ndarray, float, List[float], List[float]]:
    rng = np.random.default_rng(seed)
    buffer: Deque[Tuple[float, float, float, float]] = deque(maxlen=buffer_size)

    theta_mean = rng.normal(scale=0.1, size=2)
    log_std = np.log(0.5)
    theta_mean_target = theta_mean.copy()
    log_std_target = log_std

    critic_w1 = rng.normal(scale=0.1, size=6)
    critic_w2 = rng.normal(scale=0.1, size=6)
    critic_target_w1 = critic_w1.copy()
    critic_target_w2 = critic_w2.copy()

    log_alpha = np.log(0.2)

    returns: List[float] = []
    temperatures: List[float] = []

    for episode in range(episodes):
        state = env.reset(rng)
        episode_return = 0.0
        for t in range(horizon):
            action, mean, std, log_prob, eps = actor_forward(state, theta_mean, log_std, rng)
            next_state, reward, done = env.step(state, action, rng)
            buffer.append((state, action, reward, next_state))

            if len(buffer) >= batch_size:
                batch_idx = rng.choice(len(buffer), size=batch_size, replace=False)
                states = np.array([buffer[i][0] for i in batch_idx])
                actions = np.array([buffer[i][1] for i in batch_idx])
                rewards = np.array([buffer[i][2] for i in batch_idx])
                next_states = np.array([buffer[i][3] for i in batch_idx])

                next_samples = [actor_forward(s, theta_mean_target, log_std_target, rng) for s in next_states]
                next_actions = np.array([sample[0] for sample in next_samples])
                log_probs_next = np.array([sample[3] for sample in next_samples])

                q1_target = np.array([critic_value(s, a, critic_target_w1) for s, a in zip(next_states, next_actions)])
                q2_target = np.array([critic_value(s, a, critic_target_w2) for s, a in zip(next_states, next_actions)])
                min_q_target = np.minimum(q1_target, q2_target)
                alpha = np.exp(log_alpha)
                targets = rewards + gamma * (min_q_target - alpha * log_probs_next)

                for s, a, target in zip(states, actions, targets):
                    phi = features(s, a)
                    critic_w1 += critic_lr * (target - critic_value(s, a, critic_w1)) * phi
                    critic_w2 += critic_lr * (target - critic_value(s, a, critic_w2)) * phi

                actor_grad = np.zeros_like(theta_mean)
                log_std_grad = 0.0
                log_prob_batch = []
                for s in states:
                    a_sample, mean_s, std_s, log_prob_s, eps_s = actor_forward(s, theta_mean, log_std, rng)
                    log_prob_batch.append(log_prob_s)
                    grad_q1 = critic_grad_action(s, a_sample, critic_w1)
                    grad_q2 = critic_grad_action(s, a_sample, critic_w2)
                    grad_q = (grad_q1 + grad_q2) / 2.0

                    dlogp_dmean = (mean_s - a_sample) / (std_s ** 2)
                    dlogp_dlogstd = -1.0 + ((a_sample - mean_s) ** 2) / (std_s ** 2)

                    dmean_dtheta = np.array([s, 1.0])
                    da_dtheta = dmean_dtheta
                    da_dlogstd = std_s * eps_s

                    actor_grad += (alpha * dlogp_dmean - grad_q) * dmean_dtheta
                    log_std_grad += alpha * dlogp_dlogstd - grad_q * da_dlogstd

                actor_grad /= batch_size
                log_std_grad /= batch_size

                theta_mean += actor_lr * actor_grad
                log_std += actor_lr * np.clip(log_std_grad, -2.0, 2.0)
                log_std = np.clip(log_std, np.log(0.05), np.log(1.5))

                log_prob_batch = np.array(log_prob_batch)
                alpha = np.exp(log_alpha)
                log_alpha += alpha_lr * (alpha * (-log_prob_batch - target_entropy)).mean()

                soft_update(theta_mean_target, theta_mean, tau)
                log_std_target = (1 - tau) * log_std_target + tau * log_std
                soft_update(critic_target_w1, critic_w1, tau)
                soft_update(critic_target_w2, critic_w2, tau)

                temperatures.append(float(np.exp(log_alpha)))

            state = next_state
            episode_return += reward
            if done:
                break
        returns.append(episode_return)

    return theta_mean, log_std, returns, temperatures


def plot_returns(returns: List[float], output_path: str) -> None:
    window = 20
    if len(returns) >= window:
        running = np.convolve(returns, np.ones(window) / window, mode="valid")
    else:
        running = np.array(returns)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(returns, alpha=0.35, label="Episode return")
    ax.plot(np.arange(len(running)) + window - 1, running, color="#17becf", linewidth=1.8, label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("SAC episode returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_temperature(temperatures: List[float], output_path: str) -> None:
    if not temperatures:
        temperatures = [np.exp(np.log(0.2))]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(temperatures, color="#ff7f0e", linewidth=1.6)
    ax.set_xlabel("Update step")
    ax.set_ylabel("Temperature (alpha)")
    ax.set_title("SAC entropy temperature")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    env = SimpleContinuousEnv()
    theta, log_std, returns, temperatures = run_sac(env)

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))
    plot_temperature(temperatures, os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()