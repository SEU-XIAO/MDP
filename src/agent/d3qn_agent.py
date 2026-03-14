from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.models.d3qn import DuelingDQN
from src.replay.per_buffer import PrioritizedReplayBuffer


@dataclass
class AgentConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_update_interval: int = 500
    batch_size: int = 64
    grad_clip: float = 10.0
    per_epsilon: float = 1e-6


class D3QNAgent:
    def __init__(
        self,
        state_shape: tuple[int, int, int],
        num_actions: int,
        replay_buffer: PrioritizedReplayBuffer,
        config: AgentConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.replay_buffer = replay_buffer
        self.num_actions = num_actions

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = state_shape[0]

        self.online_net = DuelingDQN(in_channels, num_actions).to(self.device)
        self.target_net = DuelingDQN(in_channels, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.learn_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
        heuristic_prob: float = 0.0,
        heuristic_fn: Callable[[], int] | None = None,
    ) -> int:
        if random.random() < epsilon:
            if heuristic_fn is not None and random.random() < heuristic_prob:
                return heuristic_fn()
            return random.randrange(self.num_actions)

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self, beta: float) -> dict[str, float]:
        batch = self.replay_buffer.sample(self.config.batch_size, beta=beta)

        states = torch.from_numpy(batch.states).to(self.device)
        actions = torch.from_numpy(batch.actions).long().to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(batch.rewards).to(self.device).unsqueeze(1)
        next_states = torch.from_numpy(batch.next_states).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device).unsqueeze(1)
        weights = torch.from_numpy(batch.weights).to(self.device).unsqueeze(1)

        current_q = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q

        td_error = target_q - current_q
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        new_priorities = td_error.detach().abs().squeeze(1).cpu().numpy() + self.config.per_epsilon
        self.replay_buffer.update_priorities(batch.indices, new_priorities)

        self.learn_steps += 1
        if self.learn_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": float(loss.item()),
            "q_mean": float(current_q.mean().item()),
            "td_abs_mean": float(np.abs(new_priorities).mean()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        try:
            # Newer PyTorch recommends loading tensor weights only.
            state_dict = torch.load(Path(path), map_location=self.device, weights_only=True)
        except TypeError:
            # Backward compatibility for older PyTorch versions.
            state_dict = torch.load(Path(path), map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
