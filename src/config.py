from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    episodes: int = 1200
    max_steps_per_episode: int = 220
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    min_replay_size: int = 2_000
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    heuristic_start: float = 0.35
    heuristic_end: float = 0.10
    heuristic_decay_steps: int = 80_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 120_000
    risk_lambda: float = 8.0
    guide_omega: float = 1.2
    goal_reward: float = 180.0
    step_penalty: float = -1.0
    timeout_penalty: float = -60.0
    enemy_jitter: int = 0
    start_jitter: int = 0
    eval_interval_episodes: int = 50
    eval_episodes: int = 30
    eval_enemy_jitter: int = 0
    eval_start_jitter: int = 0
    observation_size: int = 64
    seed: int = 42


def load_scenario(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
