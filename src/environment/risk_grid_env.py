from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RewardWeights:
    goal_reward: float = 100.0
    step_penalty: float = -1.0
    risk_lambda: float = 5.0
    guide_omega: float = 0.7
    timeout_penalty: float = -60.0
    blocked_move_penalty: float = -8.0


class RiskAwareGridEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    ACTIONS = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    def __init__(
        self,
        scenario: dict[str, Any],
        observation_size: int = 64,
        max_steps: int = 220,
        enemy_jitter: int = 2,
        start_jitter: int = 2,
        reward_weights: RewardWeights | None = None,
        goal_sigma: float = 10.0,
        blocked_risk_threshold: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.world_size = int(scenario["map"]["grid_size"])
        self.start_pos = tuple(scenario["map"]["start_pos"])
        self.goal_pos = tuple(scenario["map"]["goal_pos"])
        self.base_enemies = scenario["enemies"]

        self.observation_size = observation_size
        self.max_steps = max_steps
        self.enemy_jitter = enemy_jitter
        self.start_jitter = start_jitter
        self.goal_sigma = goal_sigma
        self.blocked_risk_threshold = blocked_risk_threshold
        self.reward_weights = reward_weights or RewardWeights()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, observation_size, observation_size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.agent_pos = self.start_pos
        self.current_enemies = self.base_enemies
        self.step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.step_count = 0
        self.current_enemies = self._randomize_enemies(self.base_enemies)
        self.agent_pos = self._randomize_start(self.start_pos)
        while self.agent_pos == self.goal_pos:
            self.agent_pos = self._randomize_start(self.start_pos)

        obs = self._build_observation(self.agent_pos)
        info = {"risk": self._risk_at(self.agent_pos), "distance": self._distance_to_goal(self.agent_pos)}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        prev_pos = self.agent_pos
        prev_dist = self._distance_to_goal(prev_pos)

        dx, dy = self.ACTIONS[action]
        nx = int(np.clip(prev_pos[0] + dx, 0, self.world_size - 1))
        ny = int(np.clip(prev_pos[1] + dy, 0, self.world_size - 1))
        blocked = self._is_blocked_by_risk((nx, ny))
        if blocked:
            self.agent_pos = prev_pos
        else:
            self.agent_pos = (nx, ny)

        curr_dist = self._distance_to_goal(self.agent_pos)
        risk_prob = self._risk_at(self.agent_pos)

        r_goal = self.reward_weights.goal_reward if self.agent_pos == self.goal_pos else 0.0
        r_step = self.reward_weights.step_penalty
        r_risk = -self.reward_weights.risk_lambda * (risk_prob**2)
        r_guide = self.reward_weights.guide_omega * (prev_dist - curr_dist)
        r_timeout = 0.0
        r_blocked = self.reward_weights.blocked_move_penalty if blocked else 0.0

        terminated = self.agent_pos == self.goal_pos
        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            r_timeout = self.reward_weights.timeout_penalty

        reward = r_goal + r_step + r_risk + r_guide + r_timeout + r_blocked

        obs = self._build_observation(self.agent_pos)
        info = {
            "risk": risk_prob,
            "distance": curr_dist,
            "reward_breakdown": {
                "goal": r_goal,
                "step": r_step,
                "risk": r_risk,
                "guide": r_guide,
                "timeout": r_timeout,
                "blocked": r_blocked,
            },
        }
        return obs, float(reward), terminated, truncated, info

    def _is_blocked_by_risk(self, pos: tuple[int, int]) -> bool:
        if self.blocked_risk_threshold is None:
            return False
        return self._risk_at(pos) >= float(self.blocked_risk_threshold)

    def heuristic_action(self) -> int:
        best_action = 0
        best_dist = float("inf")
        for i, (dx, dy) in enumerate(self.ACTIONS):
            nx = int(np.clip(self.agent_pos[0] + dx, 0, self.world_size - 1))
            ny = int(np.clip(self.agent_pos[1] + dy, 0, self.world_size - 1))
            dist = self._distance_to_goal((nx, ny))
            if dist < best_dist:
                best_dist = dist
                best_action = i
        return best_action

    def _randomize_enemies(self, enemies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for enemy in enemies:
            px, py = enemy["pos"]
            jx = self.rng.randint(-self.enemy_jitter, self.enemy_jitter)
            jy = self.rng.randint(-self.enemy_jitter, self.enemy_jitter)
            cx = int(np.clip(px + jx, 0, self.world_size - 1))
            cy = int(np.clip(py + jy, 0, self.world_size - 1))
            new_enemy = dict(enemy)
            new_enemy["pos"] = [cx, cy]
            out.append(new_enemy)
        return out

    def _randomize_start(self, start_pos: tuple[int, int]) -> tuple[int, int]:
        sx, sy = start_pos
        jx = self.rng.randint(-self.start_jitter, self.start_jitter)
        jy = self.rng.randint(-self.start_jitter, self.start_jitter)
        nx = int(np.clip(sx + jx, 0, self.world_size - 1))
        ny = int(np.clip(sy + jy, 0, self.world_size - 1))
        return nx, ny

    def _distance_to_goal(self, pos: tuple[int, int]) -> float:
        return math.dist(pos, self.goal_pos)

    def _risk_at(self, pos: tuple[int, int]) -> float:
        px, py = pos
        survival = 1.0
        for enemy in self.current_enemies:
            ex, ey = enemy["pos"]
            dist = math.dist((px, py), (ex, ey))
            pk = self._enemy_risk_at_distance(dist, enemy["detection_zones"])
            survival *= 1.0 - pk
        return float(1.0 - survival)

    @staticmethod
    def _enemy_risk_at_distance(distance: float, zones: list[dict[str, float]]) -> float:
        sorted_zones = sorted(zones, key=lambda z: z["r"])
        for zone in sorted_zones:
            if distance <= zone["r"]:
                return float(zone["p"])
        return 0.0

    def _world_to_obs(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = pos
        if self.world_size == 1:
            return 0, 0
        ox = int(round(x * (self.observation_size - 1) / (self.world_size - 1)))
        oy = int(round(y * (self.observation_size - 1) / (self.world_size - 1)))
        return ox, oy

    def _build_observation(self, agent_pos: tuple[int, int]) -> np.ndarray:
        risk_map = self._build_risk_map()
        goal_map = self._build_goal_map()
        agent_map = np.zeros((self.observation_size, self.observation_size), dtype=np.float32)
        ax, ay = self._world_to_obs(agent_pos)
        agent_map[ay, ax] = 1.0
        return np.stack([risk_map, goal_map, agent_map], axis=0).astype(np.float32)

    def _build_risk_map(self) -> np.ndarray:
        h = self.observation_size
        w = self.observation_size
        xs = np.linspace(0, self.world_size - 1, w)
        ys = np.linspace(0, self.world_size - 1, h)
        xx, yy = np.meshgrid(xs, ys)

        survival = np.ones((h, w), dtype=np.float32)
        for enemy in self.current_enemies:
            ex, ey = enemy["pos"]
            dist = np.sqrt((xx - ex) ** 2 + (yy - ey) ** 2)
            pk = np.zeros_like(dist, dtype=np.float32)
            for zone in sorted(enemy["detection_zones"], key=lambda z: z["r"]):
                mask = (pk == 0.0) & (dist <= zone["r"])
                pk[mask] = float(zone["p"])
            survival *= 1.0 - pk
        return 1.0 - survival

    def _build_goal_map(self) -> np.ndarray:
        gx, gy = self.goal_pos
        goal_x, goal_y = self._world_to_obs((gx, gy))
        xs = np.arange(self.observation_size)
        ys = np.arange(self.observation_size)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt((xx - goal_x) ** 2 + (yy - goal_y) ** 2)
        goal_map = np.exp(-dist / self.goal_sigma)
        return goal_map.astype(np.float32)
