from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0

        self.states: list[np.ndarray | None] = [None] * capacity
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states: list[np.ndarray | None] = [None] * capacity
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.pos] = state.astype(np.float32, copy=False)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state.astype(np.float32, copy=False)
        self.dones[self.pos] = float(done)

        max_priority = self.priorities[: self.size].max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        scaled_priorities = self.priorities[: self.size] ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        states = np.stack([self.states[i] for i in indices if self.states[i] is not None], axis=0)
        next_states = np.stack(
            [self.next_states[i] for i in indices if self.next_states[i] is not None], axis=0
        )

        return TransitionBatch(
            states=states,
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=next_states,
            dones=self.dones[indices],
            indices=indices,
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities.astype(np.float32)
