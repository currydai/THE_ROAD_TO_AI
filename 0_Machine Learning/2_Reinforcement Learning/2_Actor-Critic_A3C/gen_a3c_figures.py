#!/usr/bin/env python3
"""Generate figures for the A3C tutorial."""

import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LIST = [
    "a3c_returns.png",
    "a3c_policy_heatmap.png",
]


@dataclass
class CliffWorld:
    width: int = 6
    height: int = 4
    start_state: int = 18
    goal_state: int = 23
    cliff_states: Tuple[int, ...] = tuple(range(19, 23))
    step_penalty: float = -0.02
    cliff_penalty: float = -1.0
    goal_reward: float = 1.0

    def __post_init__(self) -> None:
        self.n_states = self.width * self.height
        self.actions = [0, 1, 2, 3]  # up, right, down, left

    def reset(self) -> int:
        return self.start_state

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.width)

    def coord_to_state(self, coord: Tuple[int, int]) -> int:
        return coord[0] * self.width + coord[1]

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
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
        if next_state in self.cliff_states:
            reward = self.cliff_penalty
            next_state = self.start_state
        elif next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        return next_state, reward, done


def make_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z -= z.max()
    exp = np.exp(z)
    return exp / exp.sum()


def collect_rollout(
    env: CliffWorld,
    theta: np.ndarray,
    V: np.ndarray,
    rollout_len: int,
    gamma: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[float], int, bool]:
    states: List[int] = []
    actions: List[int] = []
    rewards: List[float] = []

    state = env.reset()
    done = False
    steps = 0

    while steps < rollout_len and not done:
        probs = softmax(theta[state])
        action = rng.choice(len(probs), p=probs)
        next_state, reward, done = env.step(state, action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        steps += 1

    return states, actions, rewards, state, done


def compute_n_step_advantages(
    states: List[int],
    rewards: List[float],
    bootstrap_state: int,
    done: bool,
    V: np.ndarray,
    gamma: float,
) -> np.ndarray:
    G = V[bootstrap_state] * (0.0 if done else 1.0)
    advantages = np.zeros(len(states))
    for t in reversed(range(len(states))):
        G = rewards[t] + gamma * G
        advantages[t] = G - V[states[t]]
    return advantages


def run_a3c(
    env: CliffWorld,
    num_workers: int = 3,
    episodes: int = 600,
    rollout_len: int = 5,
    actor_lr: float = 0.12,
    critic_lr: float = 0.1,
    gamma: float = 0.95,
    entropy_coef: float = 0.01,
    seed: int = 11,
) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    n_states = env.n_states
    n_actions = len(env.actions)

    theta_global = np.zeros((n_states, n_actions))
    V_global = np.zeros(n_states)

    returns: List[float] = []

    for episode in range(episodes):
        episode_return = 0.0
        for worker in range(num_workers):
            states, actions, rewards, bootstrap_state, done = collect_rollout(
                env, theta_global, V_global, rollout_len, gamma, rng
            )
            if not states:
                continue
            advantages = compute_n_step_advantages(
                states, rewards, bootstrap_state, done, V_global, gamma
            )

            for s, a, adv in zip(states, actions, advantages):
                td_target = adv + V_global[s]
                td_error = td_target - V_global[s]
                V_global[s] += critic_lr * adv

                probs = softmax(theta_global[s])
                grad_log = -probs
                grad_log[a] += 1.0
                theta_global[s] += actor_lr * (adv * grad_log + entropy_coef * (-np.log(probs + 1e-8) - 1.0))

            episode_return += sum(rewards)

        returns.append(episode_return / max(1, num_workers))

    return theta_global, returns


def plot_returns(returns: List[float], output_path: str) -> None:
    window = 20
    if len(returns) >= window:
        running = np.convolve(returns, np.ones(window) / window, mode="valid")
    else:
        running = np.array(returns)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(returns, alpha=0.35, label="Episode return (avg over workers)")
    ax.plot(np.arange(len(running)) + window - 1, running, color="#2ca02c", linewidth=1.8, label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("A3C aggregated returns")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_policy_heatmap(theta: np.ndarray, env: CliffWorld, output_path: str) -> None:
    probs = np.apply_along_axis(softmax, 1, theta)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(probs, cmap="viridis")
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_xlabel("Action (0=up,1=right,2=down,3=left)")
    ax.set_ylabel("State index")
    ax.set_title("Global policy probabilities after A3C training")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    figures_dir = make_output_dir()
    env = CliffWorld()
    theta, returns = run_a3c(env)

    plot_returns(returns, os.path.join(figures_dir, FIGURE_LIST[0]))
    plot_policy_heatmap(theta, env, os.path.join(figures_dir, FIGURE_LIST[1]))

    for name in FIGURE_LIST:
        print(os.path.join(figures_dir, name))


if __name__ == "__main__":
    main()